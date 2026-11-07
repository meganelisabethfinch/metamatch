from reconciliation.reconciliation_service import Entity

from .dataset import Dataset, View


class GradedPieces(Dataset):
    # One record per syllabus entry: a piece set at a grade by an exam board (ABRSM, AMEB, LCME, MTB, RCM, RIAM, Trinity).
    # Loaded from datasets/graded_pieces/works.csv with GradedPieces.from_csv. That csv was cleaned once from the
    # raw PDF-parser output (raw files kept in linked-difficulty-data/graded_pieces); preprocess.py only adds name parts.
    #
    # Columns of the clean csv:
    #   id                         row id (0, 1, 2, ...), assigned once in this csv
    #   piece_id                   the original pipeline's content hash, as used by e.g. the eval ground truth
    #                              (graded_pieces_eval.csv); shared by rows differing only in section / list
    #   piece_title, movement, work, book, grade, list, list_label, section, location, number_in_book,
    #   page_number, note, voice_type, editor, transcriber, suitable_for_lever_harp, exam_board, source_file
    #   instrument, instrument_family  canonical instrument, e.g. "horn, french" / "brass" (as written if several)
    #   instruments, instrument_families   JSON lists, one per instrument the entry is for, e.g. RIAM's
    #                              "alto/baritone saxophone" -> ["saxophone, alto", "saxophone, baritone"]
    #   composer, arranger         the name strings exactly as parsed (e.g. "Bach, J. S.", "Gilbert & Sullivan")
    #   composer_parts, arranger_parts   JSON: one NameParts per person / group / tradition (LLM name splitter)
    #   publisher_raw              the publisher exactly as parsed (e.g. RCM's "SCH; PET")
    #   publisher, publisher_code, publication_id   one publisher per row, codes expanded to full names
    #   catalogue_numbers          JSON: [{system, value}], e.g. [{"system": "BWV", "value": "BWV 1006"}]
    #   duplicate_id, is_primary, duplicate_type   the same piece in other publications / arrangements

    name = "graded_pieces"
    json_columns = ("composer_parts", "arranger_parts", "catalogue_numbers", "instruments", "instrument_families")


def _credit_ids(record: dict) -> list[str]:
    # one credit per person named in the composer field (groups / traditional left out): "<row id>/<position>"
    return [f"{record['id']}/{i}" for i, p in enumerate(record["composer_parts"] or [])
            if p["entity_type"] == "person" and p["substring"].strip()]


def _credit_entity(id: str, records: list[dict]) -> Entity:
    p = records[0]["composer_parts"][int(id.rsplit("/", 1)[1])]
    return Entity(id, p["substring"], {"original": p["substring"], "family_name": p["family_name"],
                                       "given_names": p["given_names"], "given_initials": p["given_initials"]})


def _work_descriptor(r: dict) -> str:
    # the labelled layout of the pairwise match prompts, catalogue numbers folded into the title (e.g. "Aria, BWV 515/a");
    # empty fields, and a work / book that repeats the title, left out
    title = r["piece_title"].strip()
    catalogue = ", ".join(dict.fromkeys(c["value"] for c in r["catalogue_numbers"] or [] if c.get("value")))
    lines = [f"piece_title: {f'{title}, {catalogue}' if catalogue else title}"]
    if r["movement"].strip():
        lines.append(f"movement: {r['movement'].strip()}")
    for field in ("work", "book"):
        if r[field].strip() and r[field].strip() != title:
            lines.append(f"{field}: {r[field].strip()}")
    return "\n".join(lines)


def _work_embed_text(r: dict) -> str:
    # plain text for embedding (no prompt labels): "title, catalogue numbers (from work or book)"
    title = r["piece_title"].strip()
    catalogue = ", ".join(c["value"] for c in r["catalogue_numbers"] or [])
    text = f"{title}, {catalogue}" if catalogue else title
    container = r["work"].strip() or r["book"].strip()
    return f"{text} (from {container})" if container and container != title else text


def _work_entity(id: str, records: list[dict]) -> Entity:
    r = records[0]
    properties = {k: r[k] for k in ("piece_id", "piece_title", "movement", "work", "book", "catalogue_numbers")}
    return Entity(id, _work_descriptor(r), properties | {"composers": _credit_ids(r), "embed_text": _work_embed_text(r)})


def _works_view(ds: GradedPieces, primary_only: bool = False) -> View:
    # primary_only: one row per group of duplicates (the same piece in other publications / arrangements)
    return View(ds, key=lambda r: r["id"] if not primary_only or r["is_primary"] == "True" else None,
                to_entity=_work_entity, references={"composers": "graded_pieces.composer_credits"})


GradedPieces.views = {
    # composer_credits: one entity per person named in a row's composer field, e.g. "SCHUMANN" on row 41597 is
    #   "41597/0". The value as written, not the person: two "SCHUMANN" credits may be different people (Clara /
    #   Robert), so each is reconciled (and can be disambiguated) on its own. text = the name as written,
    #   properties = family_name / given_names / given_initials, as the name verifiers need
    "composer_credits": lambda ds: View(ds, key=_credit_ids, to_entity=_credit_entity),
    # works: one entity per row (rows that differ only in e.g. book or arranger stay separate); text = the labelled
    #   descriptor (piece_title / movement / work / book) the pairwise prompts use; its "composers" property
    #   references its composer credits. primary_only=True keeps one row per group of duplicates (is_primary)
    "works": _works_view,
}
