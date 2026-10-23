# the entire pipeline:

# 1. builds/cleans the datasets
```preprocess.py [args]```
# b) plots the datasets (counts by instrument, exam board, grade / difficulty, composer) into out/plots/datasets/
```plot.py datasets [dataset ...]```   (or ```preprocess.py [args] --plot```)

# 2. runs composer reconciliation
# a) graded_pieces.composers -> music_brainz.artists | graded_pieces.composers -> wikidata.entities | imslp.composers -> music_brainz.artists | imslp.composers -> wikidata.entities | and also [any other dataset e.g. pedro ramoneda's difficulty datasets].composers -> wikidata.entities or musicbrainz.artists [no dependencies between these]
```reconcile.py [args]```
# b) runs statistics on the above e.g. reconciliation rate, agreement rate between the two target datasets, etc.
```plot.py reconciliation [job ...]```   (or ```reconcile.py [args] --plot```)

# x. run work reconciliation
# a) graded_pieces.works -> music_brainz.works | graded_pieces.works -> imslp.works | imslp.works -> music_brainz.works
``` reconcile.py [args]```
#    the LLM is chosen by name from config/paths.yaml models.llm (default qwen35-9b), one or several to compare:
```reconcile.py gp_works_mb --model qwen35-4b qwen35-9b```   (remote models also need ```--allow-remote```: they cost money)
# b) evaluate against the hand labels (datasets/eval/): the whole job on just the labelled queries, then the LLM
#    judge alone on balanced labelled pairs; scores go to out/eval/
```reconcile.py gp_works_mb --eval```   then   ```evaluate.py pipeline gp_works_mb```
```evaluate.py llm gp_works_mb```
```plot.py eval [job ...]```

# x. run cross-corpus queries
# a) for each entry in graded_pieces.works, find the corresponding entry in imslp.works (if any)
#    by following the link graded_pieces.works -> music_brainz.works -> imslp.works
# b) hence output dataset of 'difficulty-labelled imslp works' to csv.
#    [at a LATER stage of the work, we will look at segmenting imslp pdfs into parts/individual pieces].
``` build_metamatch_difficulty.py [args] ```

# c) OR run your own query by multistage linking
``` query.py [args]```

****

# to run the whole pipeline in one go:
``` main.py ```