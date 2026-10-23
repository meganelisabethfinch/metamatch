WD_HUMAN    = "Q5"       # human
WD_COMPOSER = "Q36834"   # composer

P_INSTANCE_OF    = "P31"    # instance of
P_OCCUPATION     = "P106"   # occupation
P_SUBCLASS_OF    = "P279"   # subclass of
P_FAMILY_NAME    = "P734"   # family name
P_GIVEN_NAME     = "P735"   # given name
P_PSEUDONYM      = "P742"   # pseudonym
P_INSTRUMENT     = "P1303"  # instrument
P_SERIES_ORDINAL = "P1545"  # series ordinal

MUSIC_OCCUPATION_QIDS: frozenset[str] = frozenset({
    "Q639669",    # musician
    "Q855091",    # librettist
    "Q8178443",   # librettist (alternate QID)
    "Q14915627",  # musicologist
    "Q36834",     # composer
    "Q16145150",  # music educator
    "Q822146",    # lyricist
    "Q1350157",   # music critic
    "Q61996872",  # musicographer
    "Q1643514",   # music arranger
    "Q1415090",   # film score composer
    "Q1198887",   # music director
    "Q753110",    # songwriter
    "Q158852",    # conductor
    "Q1278335",   # instrumentalist
    "Q1371914",   # soloist
    "Q10399597",  # accompanist
    "Q15981151",  # jazz musician
    "Q177220",    # singer
    "Q2643890",   # vocalist
    "Q486748",    # pianist
    "Q765778",    # organist
    "Q5371902",   # harpsichordist
    "Q11308619",  # synthesizer player
    "Q1259917",   # violinist
    "Q13219637",  # cellist
    "Q3560496",   # fiddler
    "Q6168364",   # jazz guitarist
    "Q12377274",  # trumpeter
    "Q386854",    # drummer
})

NOTABLE_OCCUPATION_QIDS: frozenset[str] = frozenset({
    "Q36180",    # writer
    "Q49757",    # poet
    "Q214917",   # playwright
    "Q6625963",  # novelist
    "Q28389",    # screenwriter
    "Q11774202", # essayist
    "Q11774156", # memoirist
    "Q4853732",  # children's writer
    "Q1930187",  # journalist
    "Q483501",   # artist
    "Q1028181",  # painter
    "Q1281618",  # sculptor
    "Q42973",    # architect
    "Q33231",    # photographer
    "Q644687",   # illustrator
    "Q33999",    # actor
    "Q2526255",  # film director
    "Q3455803",  # director
    "Q116",      # monarch
    "Q12097",    # king
    "Q82955",    # politician
    "Q189290",   # military officer
    "Q42603",    # priest
    "Q29182",    # bishop
    "Q19546",    # pope
    "Q45722",    # cardinal
    "Q43115",    # saint
    "Q1234713",  # theologian
    "Q733786",   # monk
    "Q1622272",  # university teacher
    "Q4964182",  # philosopher
    "Q201788",   # historian
    "Q170790",   # mathematician
    "Q901",      # scientist
    "Q39631",    # physician
    "Q188094",   # economist
    "Q593644",   # chemist
    "Q169470",   # physicist
    "Q11063",    # astronomer
})

PUBLISHER_TYPE_QIDS : frozenset[str] = frozenset({
    "Q184644",     # conservatory
    "Q32178211",   # music organization
    "Q5419285",    # examination board
    "Q7168296",    # performing arts presenting organization
    "Q222836",     # music publisher (profession/role)
    "Q1917775",    # music publisher (company)
    "Q2085381",    # publishing house
    "Q1320047",    # book publisher
    "Q2608849",    # imprint
    "Q3719391",    # sheet music publisher
    "Q59825643",   # online music database
})

PUBLISHER_OCCUPATION_QIDS : frozenset[str] = frozenset({
    "Q2516866",  # publisher (person)
    "Q222836",   # music publisher (profession)
})