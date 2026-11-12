"""
AquaGrievance-TN: Water Pollution Complaint Dataset generator.

Builds a reproducible, labelled dataset of citizen water-pollution complaints
(free text + metadata) for LLM-based classification and routing research.
Complaints are synthesised from category-specific templates, slot values
(places, water bodies, observations) and realistic noise (typos, lowercase,
Tanglish fillers, missing punctuation) so that a classifier has to read the
text rather than match keywords.

Usage:
    python data/generate_dataset.py            # writes data/water_pollution_complaints.csv
    python data/generate_dataset.py --n 5000   # custom size
"""
import argparse
import random
from datetime import datetime, timedelta

import pandas as pd

SEED = 42

CATEGORIES = {
    "Industrial Effluent Discharge": "State Pollution Control Board - Industrial Monitoring Cell",
    "Sewage & Drainage Overflow": "Municipal Corporation - Sewerage & Drainage Department",
    "Drinking Water Contamination": "Water Supply & Drainage Board - Water Quality Lab",
    "Solid Waste & Plastic Dumping": "Municipal Solid Waste Management Department",
    "Agricultural Runoff": "Agriculture & Water Resources Joint Cell",
    "Oil & Chemical Spill": "District Disaster Management Authority - Hazmat Response",
    "Algal Bloom & Fish Kill": "Fisheries & Aquatic Ecology Department",
    "Groundwater Contamination": "Groundwater Authority - Hydrogeology Wing",
}

DISTRICTS = {
    "Chennai": ["Adyar River", "Cooum River", "Buckingham Canal", "Pallikaranai Marsh", "Chetpet Lake"],
    "Tiruchirappalli": ["Cauvery River", "Kudamuruti River", "Uyyakondan Canal", "Kollidam River"],
    "Coimbatore": ["Noyyal River", "Ukkadam Lake", "Singanallur Lake", "Valankulam Lake"],
    "Madurai": ["Vaigai River", "Vandiyur Tank", "Mada Kulam", "Krithumal River"],
    "Tiruppur": ["Noyyal River", "Orathupalayam Dam", "Nallar Odai"],
    "Salem": ["Thirumanimutharu River", "Mookaneri Lake", "Mettur Dam backwaters"],
    "Erode": ["Bhavani River", "Kalingarayan Canal", "Cauvery River"],
    "Vellore": ["Palar River", "Vellore Fort Moat", "Otteri Lake"],
    "Thanjavur": ["Vennar River", "Grand Anicut Canal", "Sivaganga Tank"],
    "Thoothukudi": ["Tamirabarani River", "Thoothukudi coast", "Uppar Odai"],
    "Cuddalore": ["Uppanar River", "Gadilam River", "SIPCOT Cuddalore creek"],
    "Tirunelveli": ["Tamirabarani River", "Nainar Kulam", "Pachaiyar River"],
}

AREAS = ["near the bus stand", "behind the government school", "next to the temple tank", "in ward {w}",
         "near the railway bridge", "opposite the market", "near the main bridge", "on the north bank",
         "behind the housing board colony", "near the check dam", "close to the hospital road",
         "at the bathing ghat", "near the old mill", "beside the panchayat office"]

CHANNELS = ["Web Portal", "Mobile App", "Helpline 1800", "Email", "Twitter/X", "WhatsApp"]
CHANNEL_W = [0.28, 0.30, 0.16, 0.10, 0.08, 0.08]

# ---------------------------------------------------------------- templates
# {wb} water body, {area} location, {d} district, {dur} duration, {obs} observation
T = {
    "Industrial Effluent Discharge": [
        "A {ind} unit {area} is releasing {colour} effluent directly into the {wb}. {obs}",
        "Every night after 10 pm, untreated waste water from the {ind} factory is pumped into the {wb} {area}. {obs}",
        "The {ind} company in the SIPCOT estate has a hidden pipeline draining into the {wb}. Water turned {colour} for {dur}.",
        "Dyeing units {area} are discharging coloured water into the {wb}. {obs}",
        "Foamy {colour} discharge is coming out of the {ind} plant outlet into the {wb}. {obs}",
        "Their ETP is not working, the {ind} factory is letting raw effluent into the {wb} {area}.",
        "Strong chemical smell near the {wb}; a {ind} unit is dumping process water through the storm drain. {obs}",
        "We have seen tanker lorries from a {ind} industry emptying liquid waste into the {wb} {area}.",
        "Hot water and {colour} sludge from the {ind} plant is entering the {wb}. {obs}",
        "Factory waste from the {ind} unit mixes with the {wb} every time it rains, the outlet bypasses the treatment plant.",
    ],
    "Sewage & Drainage Overflow": [
        "The underground drainage is overflowing {area} and sewage is flowing into the {wb}. {obs}",
        "Manhole {area} has been overflowing for {dur}, black sewage water is entering the {wb}.",
        "Raw sewage from the apartments is let into the {wb} through an open drain. {obs}",
        "Sewer line burst {area}; toilet waste is stagnating on the road and draining into the {wb}.",
        "The pumping station {area} is not working so sewage is being diverted into the {wb}. {obs}",
        "Unbearable stink, drainage water mixing with the {wb} {area} for {dur}. {obs}",
        "Septic tank lorries are emptying sewage into the {wb} at night {area}.",
        "Blocked storm water drain {area} is carrying household sewage straight into the {wb}.",
        "Kitchen and toilet waste water from the hotels {area} is released into the {wb}. {obs}",
        "After the rain, the drainage overflowed {area} and sewage entered the {wb}; mosquitoes everywhere.",
    ],
    "Drinking Water Contamination": [
        "Corporation tap water in {area} is {tw} for {dur}. {hobs}",
        "The drinking water supplied to our street {area} smells like sewage. {hobs}",
        "Worms found in the municipal drinking water supply {area}. {hobs}",
        "Overhead tank water {area} is {tw}; we suspect the pipeline is mixing with drainage. {hobs}",
        "Metro water supply in {area} has a strong chlorine and rotten egg smell for {dur}. {hobs}",
        "The water from the public tap {area} is {tw} and has particles floating. {hobs}",
        "Pipeline leak near the drain is contaminating our drinking water {area}. {hobs}",
        "Drinking water from the RO plant {area} tastes salty and bitter. {hobs}",
        "We are getting {tw} drinking water through the house connection {area} for {dur}. {hobs}",
        "Supply water in {area} is contaminated, it looks {tw} even after boiling. {hobs}",
    ],
    "Solid Waste & Plastic Dumping": [
        "People are dumping plastic bags and garbage into the {wb} {area}. {obs}",
        "Huge heap of garbage and plastic bottles floating on the {wb} {area} for {dur}.",
        "Hotel and market waste is thrown into the {wb} every morning {area}. {obs}",
        "Construction debris and plastic waste dumped on the banks of the {wb} {area}.",
        "The {wb} is choked with thermocol, plastic covers and coconut shells {area}. {obs}",
        "Corporation garbage lorry is unloading waste near the {wb} {area}. {obs}",
        "Meat shop waste and plastic is being thrown into the {wb} {area}, stray dogs everywhere.",
        "Burnt garbage and plastic ash is being pushed into the {wb} {area}.",
        "Festival idols, flowers and plastic decorations left in the {wb} {area} for {dur}.",
        "Medical waste like syringes and plastic packets seen dumped along the {wb} {area}. {obs}",
    ],
    "Agricultural Runoff": [
        "Pesticide sprayed on the paddy fields is draining into the {wb} {area}. {obs}",
        "Farmers are washing pesticide cans and sprayers in the {wb} {area}.",
        "Fertiliser runoff from the sugarcane fields is entering the {wb} after irrigation. {obs}",
        "Water in the {wb} {area} smells of pesticide after the fields were sprayed. {obs}",
        "Banana plantation runoff with urea and chemicals is flowing into the {wb} {area}.",
        "Poultry farm and cattle shed waste is let into the irrigation canal joining the {wb}. {obs}",
        "Weedicide used in the fields near {area} is reaching the {wb} through the field channel.",
        "Excess fertiliser from the farms is making the {wb} water green and slimy {area}.",
        "Leftover pesticide mixture is being emptied into the canal that feeds the {wb}. {obs}",
        "Runoff from the coconut groves and paddy after heavy spraying is polluting the {wb} {area}.",
    ],
    "Oil & Chemical Spill": [
        "An oil tanker overturned {area} and diesel is spreading on the {wb}. {obs}",
        "Thick oil layer seen on the surface of the {wb} {area}. {obs}",
        "A chemical drum fell from a lorry {area} and the liquid is leaking into the {wb}. {obs}",
        "Petrol bunk underground tank leak, fuel is seeping into the {wb} {area}.",
        "Black oily film and dead birds on the {wb} {area} since {dur}.",
        "Pipeline carrying crude oil is leaking near the {wb} {area}. {obs}",
        "Acid tanker leakage {area}, fumes and yellow liquid entering the {wb}. {obs}",
        "Boats are leaking engine oil and used oil is being dumped into the {wb} {area}.",
        "A rainbow coloured oil sheen covers the {wb} {area}; strong kerosene smell. {obs}",
        "Unknown chemical spill near the {wb} {area}, the water is bubbling and burning eyes. {obs}",
    ],
    "Algal Bloom & Fish Kill": [
        "Hundreds of dead fish are floating in the {wb} {area}. {obs}",
        "The {wb} {area} has turned bright green with thick algae for {dur}.",
        "Fish are dying in large numbers in the {wb}; the water looks green and smells rotten. {obs}",
        "Green scum and water hyacinth have covered the entire {wb} {area}. {obs}",
        "Dead fish washed up on the banks of the {wb} {area} this morning. {obs}",
        "The {wb} has a thick green layer of algae and fish are gasping at the surface {area}.",
        "Foul smell and dead fish in the {wb} {area}, fishermen have stopped fishing.",
        "Blue green algae bloom in the {wb} {area}; cattle drinking there fell sick.",
        "Sudden fish kill in the {wb} {area}, more than {k} kg of fish dead. {obs}",
        "Water hyacinth and algae have choked the {wb} {area}, no oxygen for fish. {obs}",
    ],
    "Groundwater Contamination": [
        "Our borewell water {area} has turned {tw} for {dur}. {hobs}",
        "Open well water {area} is smelling of chemicals, we think factory waste is seeping underground. {hobs}",
        "Borewell water in {area} is very salty and hard; tests show high TDS. {hobs}",
        "Hand pump water {area} is yellowish and has an oily smell. {hobs}",
        "High fluoride in our borewell {area}, children have yellow teeth. {hobs}",
        "Well water in our village {area} has become {tw} after the tannery started nearby. {hobs}",
        "Borewell water {area} turned red brown with iron and leaves stains on vessels. {hobs}",
        "Groundwater near the dump yard {area} is contaminated, the borewell smells like leachate. {hobs}",
        "All the borewells {area} give foul smelling water since {dur}. {hobs}",
        "Well water {area} is unfit for use, nitrate levels are high according to the lab. {hobs}",
    ],
}

SLOTS = {
    "ind": ["dyeing", "tannery", "textile", "paper mill", "sugar mill", "chemical", "electroplating",
            "pharmaceutical", "distillery", "bleaching", "foundry", "fertiliser"],
    "colour": ["red", "blue", "dark brown", "black", "violet", "yellow", "milky white", "greenish"],
    "tw": ["muddy", "yellowish", "smelly", "brownish", "milky", "oily", "greenish", "foul smelling"],
    "dur": ["two days", "three days", "a week", "two weeks", "a month", "the last 10 days", "four days", "yesterday"],
    "k": ["50", "100", "200", "500"],
}

# severity -> phrases (priority signal)
OBS = {
    "High": ["Children are falling sick after contact with the water.", "Cattle that drank the water have died.",
             "People are getting skin rashes and vomiting.", "This is an emergency, please act immediately.",
             "Many families depend on this water and are in danger.", "Fish and birds are dying.",
             "The fumes are causing breathing problems for residents."],
    "Medium": ["The smell is unbearable and spreading to houses.", "This has been going on for many days.",
               "Residents are worried about mosquitoes and diseases.", "The water colour has changed completely.",
               "Nobody is taking action despite earlier complaints.", "It is getting worse every day."],
    "Low": ["Please look into this when possible.", "Kindly check this.", "Just bringing this to your notice.",
            "It is a small area but should be cleaned.", "Requesting an inspection.", ""],
}
HOBS = {
    "High": ["Many people in our street have diarrhoea and fever.", "My children were hospitalised after drinking it.",
             "Elderly people are falling sick, urgent action needed.", "Several cases of jaundice reported in our area."],
    "Medium": ["We are forced to buy can water.", "We cannot use it for cooking or bathing.",
               "Please test the water quality.", "We have complained twice before."],
    "Low": ["Kindly check the supply.", "Please look into this.", "Requesting a water test.", ""],
}

OPENERS = ["", "", "", "Sir, ", "Respected officer, ", "Dear sir/madam, ", "Hello, ", "Complaint: ", "Urgent: ",
           "Sir please, ", "Good morning, "]
CLOSERS = ["", "", "", " Please take action.", " Kindly do the needful.", " Thank you.", " Pls help.",
           " Waiting for your response.", " Romba kashtama irukku sir.", " Photos attached."]

PRIORITY_W = {
    "Oil & Chemical Spill": [0.65, 0.25, 0.10],
    "Industrial Effluent Discharge": [0.40, 0.42, 0.18],
    "Drinking Water Contamination": [0.50, 0.35, 0.15],
    "Algal Bloom & Fish Kill": [0.40, 0.40, 0.20],
    "Sewage & Drainage Overflow": [0.25, 0.50, 0.25],
    "Groundwater Contamination": [0.30, 0.45, 0.25],
    "Agricultural Runoff": [0.20, 0.45, 0.35],
    "Solid Waste & Plastic Dumping": [0.12, 0.43, 0.45],
}
CAT_W = {
    "Sewage & Drainage Overflow": 0.18, "Industrial Effluent Discharge": 0.16,
    "Solid Waste & Plastic Dumping": 0.15, "Drinking Water Contamination": 0.14,
    "Groundwater Contamination": 0.10, "Algal Bloom & Fish Kill": 0.10,
    "Agricultural Runoff": 0.09, "Oil & Chemical Spill": 0.08,
}
SLA_HOURS = {"High": 24, "Medium": 72, "Low": 168}

# Mild side-mentions of OTHER problems (distractors). The label is always the main/root issue.
SIDE = {
    "Industrial Effluent Discharge": "There is also a factory chimney nearby.",
    "Sewage & Drainage Overflow": "The drains in the area also smell a little.",
    "Drinking Water Contamination": "We are also worried about our drinking water.",
    "Solid Waste & Plastic Dumping": "Some plastic covers are also lying around.",
    "Agricultural Runoff": "There are paddy fields on the other side.",
    "Oil & Chemical Spill": "A few vehicles are also parked near the bank.",
    "Algal Bloom & Fish Kill": "We noticed a few dead fish too.",
    "Groundwater Contamination": "Our borewell is also nearby.",
}
HOLDOUT_TEMPLATES = {7, 8, 9}  # unseen phrasings -> test split (measures generalisation)


def add_noise(text, rng):
    if rng.random() < 0.18:
        text = text.lower()
    if rng.random() < 0.15:
        text = text.replace(". ", " ").replace(",", "")
    if rng.random() < 0.20:  # character-level typos
        chars = list(text)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(chars))
            if chars[i].isalpha():
                chars[i] = rng.choice("aeiourstn")
        text = "".join(chars)
    if rng.random() < 0.06:
        text = text.upper()
    return text


def make_complaint(cat, rng):
    district = rng.choice(list(DISTRICTS))
    wb = rng.choice(DISTRICTS[district])
    area = rng.choice(AREAS).format(w=rng.randint(1, 65))
    priority = rng.choices(["High", "Medium", "Low"], weights=PRIORITY_W[cat])[0]
    t_idx = rng.randrange(len(T[cat]))
    tpl = T[cat][t_idx]
    fill = {k: rng.choice(v) for k, v in SLOTS.items()}
    body = tpl.format(wb=wb, area=f"{area} in {district}" if rng.random() < 0.5 else area, d=district,
                      obs=rng.choice(OBS[priority]), hobs=rng.choice(HOBS[priority]), **fill)
    # make sure the severity signal is present even if the template had no slot for it
    if "{obs}" not in tpl and "{hobs}" not in tpl and priority != "Low":
        pool = HOBS if cat in ("Drinking Water Contamination", "Groundwater Contamination") else OBS
        body += " " + rng.choice(pool[priority])
    if rng.random() < 0.25:
        other = rng.choice([c for c in SIDE if c != cat])
        body += " " + SIDE[other]
    text = (rng.choice(OPENERS) + body + rng.choice(CLOSERS)).strip()
    text = " ".join(text.split())
    split = "test" if t_idx in HOLDOUT_TEMPLATES else "train"
    return district, wb, priority, add_noise(text, rng), split


def build(n=2500, seed=SEED):
    rng = random.Random(seed)
    start = datetime(2025, 1, 1)
    cats, w = zip(*CAT_W.items())
    rows, seen = [], set()
    while len(rows) < n:
        cat = rng.choices(cats, weights=w)[0]
        district, wb, priority, text, split = make_complaint(cat, rng)
        if text in seen:
            continue
        seen.add(text)
        ts = start + timedelta(days=rng.randint(0, 669), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
        rows.append({
            "complaint_text": text,
            "category": cat,
            "department": CATEGORIES[cat],
            "priority": priority,
            "sla_hours": SLA_HOURS[priority],
            "district": district,
            "water_body": wb,
            "channel": rng.choices(CHANNELS, weights=CHANNEL_W)[0],
            "reported_at": ts.strftime("%Y-%m-%d %H:%M"),
            "split": split,
        })
    df = pd.DataFrame(rows).sort_values("reported_at", kind="stable").reset_index(drop=True)
    df.insert(0, "complaint_id", [f"WPC-{i:05d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", default="data/water_pollution_complaints.csv")
    a = ap.parse_args()
    df = build(a.n)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} complaints -> {a.out}")
    print(df.category.value_counts().to_string())
    print(df.priority.value_counts().to_string())
    print(df.split.value_counts().to_string())
