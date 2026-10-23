from __future__ import annotations

from functools import cache

from constants.wikidata import P_OCCUPATION, WD_COMPOSER, WD_HUMAN
from data.graded_pieces import GradedPieces
from data.imslp import IMSLPComposers, IMSLPWorks, for_instrumentation
from data.musicbrainz import MusicBrainzArtists, MusicBrainzWorks
from data.wikidata import WikiData
from graph.relation import IDENTITY_RELATIONS, WORK_RELATIONS, Relation
from prompts.pairwise_match import LABELS, pairwise_prompt
from utils.llm.model import is_remote, llm, llm_names
from utils.llm.pairwise_parser import PairwiseParser
from utils.names import fullest_name, is_family_name_only
from utils.paths import PATHS

from .cached_service import CachedService
from .cascade_service import CascadeService
from .embedding_service import EmbeddingService
from .guided_search_service import GuidedSearchService
from .fallback_service import FallbackService
from .job import Job
from .llm_service import LLMService
from .open_refine_service import OpenRefineService
from .property_source import ViewProperties
from .search_service import SearchService
from .verifier_service import AnyOf, VerifierService
from .verifiers import MusicOccupation, NotableOccupation, name_check
from .voting_service import VotingService


@cache
def graded_pieces(instrument=None):
    # loaded once per run (several jobs and prefilters share it); instrument: e.g. "piano", as the old work matching
    gp = GradedPieces.from_csv(PATHS["graded_pieces"]["works"])
    return gp.filter(lambda df: df.instrument == instrument) if instrument else gp


def graded_composers():
    return graded_pieces().view("composer_credits")


def imslp_composers():
    # composers credited on at least one IMSLP work (the ones whose works we'll want to match)
    return IMSLPComposers.from_dumps(PATHS["imslp"]["composers"], PATHS["imslp"]["works"]).view("composers")


# The LLM a job that takes one (Job.takes_model) uses when none is named: a key of paths.yaml models.llm
DEFAULT_LLM = "qwen35-9b"

# Vote sets: named like models (reconcile.py / evaluate.py --model): the majority answer of these LLMs, strongest first
# (a tie goes to the strongest); each LLM's answers are reused from its cache where it has given them before
VOTES = {"vote-qwen": ["qwen35-9b", "qwen35-4b", "qwen25-7b"],
         "vote-all": ["claude-opus-4-8", "claude-haiku-4-5", "qwen35-9b", "qwen35-4b", "qwen25-7b"]}


def model_names() -> list[str]:
    # what --model can name: an LLM (config/paths.yaml models.llm) or a vote set
    return [*llm_names(), *VOTES]


def uses_remote(model: str) -> bool:
    # whether the model (or any voter in a vote set) runs remotely, and so costs money
    return any(is_remote(m) for m in VOTES.get(model, [model]))

# The IMSLP works a piano job looks at: IMSLP files much early keyboard music (Bach, Scarlatti, Handel, ...) as for
# keyboard, harpsichord or organ rather than piano, though it is piano exam repertoire
IMSLP_PIANO = ("piano", "keyboard", "harpsichord", "organ")


@cache
def imslp_works(*instrumentations):
    # loaded once per run; instrumentations: only works for these, as IMSLP names them (see for_instrumentation),
    # e.g. ("piano",); none = every work
    works = IMSLPWorks.from_dumps(PATHS["imslp"]["works"], PATHS["imslp"]["works_metadata"])
    return works.filter(for_instrumentation(*instrumentations)) if instrumentations else works


def wikidata_entities():
    return WikiData.load(PATHS["wikidata"]["entities"]).view("entities")


def musicbrainz_artists():
    paths = PATHS["musicbrainz"]
    return MusicBrainzArtists(paths["artist_index"], paths["works_index"]).view("artists")


def musicbrainz_works():
    return MusicBrainzWorks(PATHS["musicbrainz"]["works_index"]).view("works")


def composers_to_wikidata(occupation):
    # OpenRefine finds up to 5 humans whose occupation is composer (score >= 70); the first that passes the
    # occupation and name checks (fetched from Wikidata) is the match.
    # Family-name-only queries with no match are retried as "<name> composer", which brings the right composer into
    # OpenRefine's candidates via their description; those scores run lower, so the threshold is 65
    def search_then_verify(threshold, query_text=None):
        return CascadeService([
            CachedService(OpenRefineService(type_id=WD_HUMAN, properties=[{"pid": P_OCCUPATION, "v": {"id": WD_COMPOSER}}],
                                            threshold=threshold, query_text=query_text),
                          PATHS["service_cache"] / "openrefine.jsonl"),
            VerifierService([occupation, name_check()], target_properties=ViewProperties(wikidata_entities()),
                            relation=Relation.IS, top_k=1),
        ])
    return FallbackService(search_then_verify(70),
                           search_then_verify(65, query_text=lambda q: f"{q.text} composer"),
                           retry_if=lambda q: is_family_name_only(q.properties))


def composers_to_musicbrainz():
    # full-text search of the local MusicBrainz index for the fullest name the parts give (e.g. "J Schubert"), with
    # name similarity >= 70; the first candidate that passes the name checks is the match.
    # (No HasWorks check: as in linked-difficulty-data, whose MusicBrainz reconciler was never given the works index)
    artists = musicbrainz_artists()
    return CascadeService([
        CachedService(SearchService(artists.dataset.search, artists, threshold=70,
                                    query_text=lambda q: fullest_name(q.properties, q.text)),
                      PATHS["service_cache"] / "musicbrainz_search.jsonl", view=artists),
        VerifierService([name_check()], relation=Relation.IS, top_k=1),
    ])


def works_service(target, prompt, model=None, tree=False, hierarchy=None, top_k=30, batch=None):
    # every works job: rank the candidate works by embedding similarity, keep the top_k, then ask the LLM (pairwise
    # prompt) about them in that order until it says IS
    # target: the target View; prompt: pairwise_prompt's arguments, (query dataset, target dataset[, style, examples]),
    #   e.g. ("graded_pieces", "musicbrainz"); tree: after a partial match, try that work's parts / parents next (needs
    #   hierarchy, the target property listing what each work is part of, e.g. "part_of"); batch: search many
    #   queries at once, as many as the model takes together (its batch_size), each round in one batch; None: the
    #   model's default (on for remote models, off for local). model: an LLM's name, or a vote set's (VOTES)
    template = pairwise_prompt(*prompt)
    judges = {m: LLMService(llm(m), template, PairwiseParser(LABELS), batch=batch)
              for m in VOTES.get(model or DEFAULT_LLM, [model or DEFAULT_LLM])}
    cached = {m: CachedService(j, PATHS["service_cache"] / "llm_pairwise.jsonl") for m, j in judges.items()}
    judge = VotingService(cached) if len(cached) > 1 else next(iter(cached.values()))
    return CascadeService([
        EmbeddingService(PATHS["models"]["embedding"], PATHS["embedding_cache"], top_k=top_k),
        GuidedSearchService(judge, target, hierarchy=hierarchy, tree=tree, batch_size=_window(judges.values())),
    ])


def _window(judges) -> int | None:
    # how many queries a search keeps open at once: as many as its judges take together (None: any number, e.g. an
    # Anthropic batch); 1 if none batches. A vote: every query, so each voter is loaded once per round, not once per
    # judgement (local models are loaded one at a time: utils.llm.model.MAX_LOADED)
    judges = list(judges)
    if len(judges) > 1:
        return None
    sizes = [j.llm.batch_size for j in judges if j.batch]
    return None if any(s is None for s in sizes) else max(sizes, default=1)


def gp_works_to_musicbrainz(model=None, tree=False):
    return works_service(musicbrainz_works(), ("graded_pieces", "musicbrainz"), model, tree=tree, hierarchy="part_of")


def gp_works_to_imslp(model=None):
    # flat: IMSLP works have no part-of hierarchy
    return works_service(imslp_works(*IMSLP_PIANO).view("works"), ("graded_pieces", "imslp"), model)


def imslp_works_to_musicbrainz(model=None, tree=False):
    # the graded-pieces prompt's examples, so the three works prompts differ only in how A and B are described
    return works_service(musicbrainz_works(), ("imslp", "musicbrainz", "direct", "graded_pieces"), model, tree=tree,
                         hierarchy="part_of")


def piano_works_to_musicbrainz_path():
    # MB works by the same composer as the graded work, via any route between their composer credit and MB artist
    return [("composers", graded_pieces("piano").view("composer_credits")), musicbrainz_artists(),
            ("composers", musicbrainz_works())]


def imslp_piano_works_to_musicbrainz_path():
    # MB works by the same composer as the IMSLP work: the MB artists its IMSLP composer is linked to, then their works
    return [("composers", imslp_composers()), musicbrainz_artists(), ("composers", musicbrainz_works())]


def piano_works_to_imslp_path():
    # IMSLP piano works by the same composer as the graded work: the IMSLP composers its composer credit is linked
    # to, by any route (Wikidata, MusicBrainz), then their works
    return [("composers", graded_pieces("piano").view("composer_credits")), imslp_composers(),
            ("composers", imslp_works(*IMSLP_PIANO).view("works"))]


JOBS = {job.name: job for job in [
    Job("gp_composers_wd", graded_composers, wikidata_entities,
        lambda: composers_to_wikidata(MusicOccupation()), IDENTITY_RELATIONS),
    # IMSLP credits are less strictly composers (e.g. writers, monarchs), so a broadly notable occupation also passes
    Job("imslp_composers_wd", imslp_composers, wikidata_entities,
        lambda: composers_to_wikidata(AnyOf([MusicOccupation(), NotableOccupation()], name="occupation",
                                            failure="no music-related or notable occupation")),
        IDENTITY_RELATIONS),
    Job("gp_composers_mb", graded_composers, musicbrainz_artists, composers_to_musicbrainz, IDENTITY_RELATIONS),
    Job("imslp_composers_mb", imslp_composers, musicbrainz_artists, composers_to_musicbrainz, IDENTITY_RELATIONS),
    # works: graded piano works -> MusicBrainz works by the same composer (needs the composer jobs first)
    Job("gp_works_mb", lambda: graded_pieces("piano").view("works"), musicbrainz_works,
        gp_works_to_musicbrainz, WORK_RELATIONS, piano_works_to_musicbrainz_path, takes_model=True),
    Job("gp_works_mb_tree", lambda: graded_pieces("piano").view("works"), musicbrainz_works,
        lambda model: gp_works_to_musicbrainz(model, tree=True), WORK_RELATIONS, piano_works_to_musicbrainz_path,
        takes_model=True),
    # graded piano works -> IMSLP keyboard works (IMSLP_PIANO) by the same composer (needs the composer jobs first)
    Job("gp_works_imslp", lambda: graded_pieces("piano").view("works"), lambda: imslp_works(*IMSLP_PIANO).view("works"),
        gp_works_to_imslp, WORK_RELATIONS, piano_works_to_imslp_path, takes_model=True),
    # IMSLP keyboard works (IMSLP_PIANO, or arranged for one) -> MusicBrainz works by the same composer: with
    # gp_works_mb, links graded pieces to IMSLP through MusicBrainz (needs the IMSLP composer jobs first)
    Job("imslp_works_mb", lambda: imslp_works(*IMSLP_PIANO).view("works"), musicbrainz_works,
        imslp_works_to_musicbrainz, WORK_RELATIONS, imslp_piano_works_to_musicbrainz_path, takes_model=True),
]}
