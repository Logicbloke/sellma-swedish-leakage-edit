"""Build all prompt / evaluation sets deterministically.

Instruction prompts: 48 topics x 5 task templates, written in parallel Swedish/English/Norwegian (Bokmål)/Danish.
Splits are by TOPIC so no topic leaks between direction-finding, dev (alpha/layer selection) and test.
Agreement pairs: en/ett article minimal pairs over nouns of known gender (test-only, never used for selection).
Perplexity text: lead sections of Swedish / English Wikipedia articles (CC BY-SA 4.0), fetched once and committed.
"""
import json, random, time, subprocess, urllib.parse, pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
SEED = 0

TOPICS = [  # (swedish, english)
    ("fotosyntes", "photosynthesis"), ("kompostering", "composting"), ("vaccination", "vaccination"),
    ("inflation", "inflation"), ("demokrati", "democracy"), ("solenergi", "solar energy"),
    ("vindkraft", "wind power"), ("sömn", "sleep"), ("motion", "exercise"), ("meditation", "meditation"),
    ("programmering", "programming"), ("artificiell intelligens", "artificial intelligence"),
    ("kryptering", "encryption"), ("elbilar", "electric cars"), ("kollektivtrafik", "public transport"),
    ("återvinning", "recycling"), ("sparande", "saving money"), ("aktiehandel", "stock trading"),
    ("distansarbete", "remote work"), ("friluftsliv", "outdoor life"), ("skogsbruk", "forestry"),
    ("fiske", "fishing"), ("matlagning", "cooking"), ("vegetarisk kost", "a vegetarian diet"),
    ("kaffe", "coffee"), ("bakning", "baking"), ("trädgårdsskötsel", "gardening"), ("läsning", "reading"),
    ("språkinlärning", "language learning"), ("schack", "chess"), ("fotboll", "football"),
    ("längdskidåkning", "cross-country skiing"), ("musik", "music"), ("fotografering", "photography"),
    ("sociala medier", "social media"), ("källkritik", "source criticism"),
    ("allemansrätten", "the right of public access"), ("midsommar", "Midsummer"), ("vulkaner", "volcanoes"),
    ("jordbävningar", "earthquakes"), ("norrsken", "the northern lights"),
    ("biologisk mångfald", "biodiversity"), ("antibiotika", "antibiotics"), ("första hjälpen", "first aid"),
    ("privatekonomi", "personal finance"), ("integritet på nätet", "online privacy"),
    ("kärnkraft", "nuclear power"), ("rymdforskning", "space research"),
]

TEMPLATES = [  # (swedish, english)
    ("Förklara kort vad {} är.", "Briefly explain what {} is."),
    ("Ge tre praktiska tips om {}.", "Give three practical tips about {}."),
    ("Vilka är för- och nackdelarna med {}?", "What are the pros and cons of {}?"),
    ("Skriv ett kort stycke om {} för en tolvåring.", "Write a short paragraph about {} for a twelve-year-old."),
    ("Varför spelar {} roll i vardagen?", "Why does {} matter in everyday life?"),
]

# Norwegian Bokmål / Danish topic names, same order as TOPICS
TOPICS_NO = ["fotosyntese", "kompostering", "vaksinasjon", "inflasjon", "demokrati", "solenergi", "vindkraft",
    "søvn", "trening", "meditasjon", "programmering", "kunstig intelligens", "kryptering", "elbiler",
    "kollektivtransport", "resirkulering", "sparing", "aksjehandel", "hjemmekontor", "friluftsliv", "skogbruk",
    "fiske", "matlaging", "vegetarisk kosthold", "kaffe", "baking", "hagearbeid", "lesing", "språklæring", "sjakk",
    "fotball", "langrenn", "musikk", "fotografering", "sosiale medier", "kildekritikk", "allemannsretten",
    "sankthans", "vulkaner", "jordskjelv", "nordlys", "biologisk mangfold", "antibiotika", "førstehjelp",
    "privatøkonomi", "personvern på nett", "kjernekraft", "romforskning"]
TOPICS_DA = ["fotosyntese", "kompostering", "vaccination", "inflation", "demokrati", "solenergi", "vindkraft",
    "søvn", "motion", "meditation", "programmering", "kunstig intelligens", "kryptering", "elbiler",
    "offentlig transport", "genbrug", "opsparing", "aktiehandel", "hjemmearbejde", "friluftsliv", "skovbrug",
    "fiskeri", "madlavning", "vegetarisk kost", "kaffe", "bagning", "havearbejde", "læsning", "sprogindlæring",
    "skak", "fodbold", "langrend", "musik", "fotografering", "sociale medier", "kildekritik", "allemandsretten",
    "sankthans", "vulkaner", "jordskælv", "nordlys", "biodiversitet", "antibiotika", "førstehjælp",
    "privatøkonomi", "privatliv på nettet", "atomkraft", "rumforskning"]
TEMPLATES_NO = ["Forklar kort hva {} er.", "Gi tre praktiske tips om {}.", "Hva er fordelene og ulempene med {}?",
                "Skriv et kort avsnitt om {} for en tolvåring.", "Hvorfor spiller {} en rolle i hverdagen?"]
TEMPLATES_DA = ["Forklar kort, hvad {} er.", "Giv tre praktiske tips om {}.", "Hvad er fordelene og ulemperne ved {}?",
                "Skriv et kort afsnit om {} til en tolvårig.", "Hvorfor spiller {} en rolle i hverdagen?"]
assert len(TOPICS_NO) == len(TOPICS_DA) == len(TOPICS)

SPLIT_SIZES = {"dir": 20, "dev": 8, "test": 20}  # topics per split

EN_NOUNS = "bil stol katt hund lampa bok penna väska cykel båt flicka pojke telefon dator säng kopp tallrik sko jacka mössa fågel häst ko gata stad sjö flod dörr nyckel klocka blomma sten buss ö låda burk sked gaffel kniv skjorta".split()
ETT_NOUNS = "hus bord äpple barn träd fönster glas papper kök rum tåg flygplan djur får lejon ägg bröd land berg hav kort brev foto spel ljus moln fält staket slott torn".split()
FRAMES = ["Jag såg {} {} i går.", "Det finns {} {} i närheten."]

WIKI = {
    "sv": ["Stockholm", "Fotosyntes", "Sveriges_riksdag", "Vasaloppet", "Allemansrätten", "Midsommar",
           "Astrid_Lindgren", "Norrsken", "Skogsbruk", "Göta_kanal"],
    "en": ["Stockholm", "Photosynthesis", "Riksdag", "Vasaloppet", "Freedom_to_roam", "Midsummer",
           "Astrid_Lindgren", "Aurora", "Forestry", "Göta_Canal"],
}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prompts():
    rng = random.Random(SEED)
    idx = list(range(len(TOPICS)))
    rng.shuffle(idx)
    start = 0
    for split, n in SPLIT_SIZES.items():
        topics = idx[start:start + n]
        start += n
        fill = {"sv": lambda t, k: TEMPLATES[k][0].format(TOPICS[t][0]),
                "en": lambda t, k: TEMPLATES[k][1].format(TOPICS[t][1]),
                "no": lambda t, k: TEMPLATES_NO[k].format(TOPICS_NO[t]),
                "da": lambda t, k: TEMPLATES_DA[k].format(TOPICS_DA[t])}
        for lang, f in fill.items():
            rows = [{"id": f"{split}-{t}-{k}", "topic": TOPICS[t][1], "lang": lang, "prompt": f(t, k)}
                    for t in topics for k in range(len(TEMPLATES))]
            write_jsonl(DATA / f"{split}_{lang}.jsonl", rows)


def agreement():
    rows = []
    for gender, nouns in (("en", EN_NOUNS), ("ett", ETT_NOUNS)):
        wrong = "ett" if gender == "en" else "en"
        for n in nouns:
            for fi, fr in enumerate(FRAMES):
                rows.append({"noun": n, "gender": gender, "frame": fi,
                             "good": fr.format(gender, n), "bad": fr.format(wrong, n)})
    write_jsonl(DATA / "agreement_pairs.jsonl", rows)


def wiki():
    # curl rather than urllib: Wikimedia rate-limits Python's urllib client fingerprint.
    for lang, titles in WIKI.items():
        if (DATA / f"ppl_{lang}.jsonl").exists():
            continue
        paras = []
        for t in titles:
            time.sleep(3)
            url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1"
                   f"&redirects=1&format=json&titles={urllib.parse.quote(t)}")
            raw = subprocess.run(["curl", "-sf", "--retry", "6", "--retry-delay", "15", "-A", "sellma-drift-experiment/0.1 (research)", url],
                                 capture_output=True, check=True).stdout
            page = next(iter(json.loads(raw)["query"]["pages"].values()))
            text = " ".join(page["extract"].split())
            assert text, t
            paras.append({"title": t, "text": text[:2000], "license": "CC BY-SA 4.0",
                          "source": f"https://{lang}.wikipedia.org/wiki/{t}"})
        write_jsonl(DATA / f"ppl_{lang}.jsonl", paras)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    prompts()
    agreement()
    wiki()
    # overlap check
    seen = {}
    for split in SPLIT_SIZES:
        for r in map(json.loads, open(DATA / f"{split}_sv.jsonl")):
            assert r["topic"] not in seen or seen[r["topic"]] == split, r
            seen[r["topic"]] = split
    print("ok", {s: sum(1 for _ in open(DATA / f"{s}_sv.jsonl")) for s in SPLIT_SIZES})
