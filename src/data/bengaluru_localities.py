"""
Comprehensive Bengaluru (Bangalore) localities, coordinates, and reference points.

Covers 150+ localities across all zones:
- East: Whitefield, Marathahalli, Bellandur, Varthur, KR Puram, Hoodi, etc.
- South: Koramangala, HSR Layout, JP Nagar, Jayanagar, BTM Layout, Bannerghatta, Electronic City, Kanakapura Road, etc.
- North: Hebbal, Yelahanka, Thanisandra, Hennur, Sahakara Nagar, Vidyaranyapura, Jakkur, Devanahalli, etc.
- West: Rajajinagar, Malleshwaram, Vijayanagar, Basaveshwaranagar, Nagarbhavi, Kengeri, Yeshwanthpur, etc.
- Central: MG Road, Indiranagar, Richmond Town, Frazer Town, Ulsoor, Domlur, Benson Town, Sadashiva Nagar, etc.
- Suburbs/Peripheral: Sarjapur, Chandapura, Anekal, Jigani, Bidadi, Nelamangala, Hoskote, Attibele, etc.
"""

from typing import Dict, Tuple, Optional
import numpy as np

# ── Comprehensive Bengaluru Localities Database ───────────────────────────────
BENGALURU_LOCALITIES: Dict[str, Tuple[float, float]] = {
    # ── Central Bengaluru
    "MG Road": (12.9756, 77.6094),
    "Brigade Road": (12.9734, 77.6074),
    "Commercial Street": (12.9822, 77.6083),
    "Richmond Town": (12.9665, 77.6068),
    "Langford Town": (12.9597, 77.6033),
    "Shanti Nagar": (12.9568, 77.5986),
    "Vasanth Nagar": (12.9902, 77.5898),
    "Cunningham Road": (12.9868, 77.5936),
    "Sadashiva Nagar": (13.0084, 77.5804),
    "Seshadripuram": (12.9932, 77.5768),
    "Ulsoor": (12.9817, 77.6286),
    "Domlur": (12.9609, 77.6387),
    "Indiranagar": (12.9784, 77.6408),
    "Indiranagar 100ft Road": (12.9719, 77.6412),
    "Defence Colony Indiranagar": (12.9750, 77.6450),
    "Frazer Town": (12.9988, 77.6126),
    "Cooke Town": (13.0012, 77.6200),
    "Cox Town": (12.9972, 77.6231),
    "Benson Town": (13.0016, 77.5997),
    "Richards Town": (13.0042, 77.6167),
    "Austin Town": (12.9631, 77.6189),
    "Victoria Layout": (12.9658, 77.6150),

    # ── South Bengaluru & Tech Corridor
    "Koramangala": (12.9352, 77.6245),
    "Koramangala 1st Block": (12.9260, 77.6360),
    "Koramangala 3rd Block": (12.9300, 77.6280),
    "Koramangala 4th Block": (12.9340, 77.6310),
    "Koramangala 5th Block": (12.9350, 77.6190),
    "Koramangala 6th Block": (12.9390, 77.6240),
    "HSR Layout": (12.9121, 77.6446),
    "HSR Layout Sector 1": (12.9210, 77.6520),
    "HSR Layout Sector 2": (12.9150, 77.6490),
    "HSR Layout Sector 3": (12.9120, 77.6410),
    "HSR Layout Sector 4": (12.9080, 77.6400),
    "HSR Layout Sector 7": (12.9050, 77.6470),
    "BTM Layout": (12.9166, 77.6101),
    "BTM 1st Stage": (12.9200, 77.6150),
    "BTM 2nd Stage": (12.9130, 77.6080),
    "Jayanagar": (12.9308, 77.5838),
    "Jayanagar 4th Block": (12.9290, 77.5830),
    "Jayanagar 9th Block": (12.9180, 77.5930),
    "JP Nagar": (12.9063, 77.5857),
    "JP Nagar Phase 1": (12.9140, 77.5900),
    "JP Nagar Phase 2": (12.9090, 77.5910),
    "JP Nagar Phase 5": (12.9040, 77.5880),
    "JP Nagar Phase 7": (12.8950, 77.5790),
    "JP Nagar Phase 8": (12.8820, 77.5720),
    "Banashankari": (12.9255, 77.5468),
    "Banashankari 2nd Stage": (12.9280, 77.5600),
    "Banashankari 3rd Stage": (12.9220, 77.5400),
    "Banashankari 6th Stage": (12.8750, 77.5250),
    "Basavanagudi": (12.9422, 77.5753),
    "Padmanabhanagar": (12.9180, 77.5580),
    "Kumaraswamy Layout": (12.9056, 77.5567),
    "Uttarahalli": (12.9055, 77.5140),
    "Girinagar": (12.9430, 77.5400),
    "Hanumanthnagar": (12.9470, 77.5620),
    "ISRO Layout": (12.8980, 77.5610),

    # ── South Peripheral & Bannerghatta / Kanakapura / Electronic City
    "Bannerghatta Road": (12.8913, 77.5982),
    "Arekere": (12.8860, 77.5960),
    "Hulimavu": (12.8785, 77.5960),
    "Gottigere": (12.8596, 77.5880),
    "Bilekahalli": (12.8980, 77.6040),
    "Bommanahalli": (12.9089, 77.6240),
    "Begur": (12.8797, 77.6247),
    "Begur Road": (12.8850, 77.6300),
    "Kudlu": (12.8910, 77.6520),
    "Kudlu Gate": (12.8893, 77.6493),
    "Singasandra": (12.8790, 77.6440),
    "Hosa Road": (12.8750, 77.6550),
    "Electronic City": (12.8399, 77.6770),
    "Electronic City Phase 1": (12.8450, 77.6630),
    "Electronic City Phase 2": (12.8452, 77.6890),
    "Neeladri Nagar Electronic City": (12.8430, 77.6610),
    "Bommasandra": (12.8167, 77.6833),
    "Bommasandra Industrial Area": (12.8120, 77.6900),
    "Chandapura": (12.7937, 77.7011),
    "Anekal": (12.7107, 77.6974),
    "Jigani": (12.7844, 77.6394),
    "Jigani Industrial Area": (12.7780, 77.6350),
    "Attibele": (12.7800, 77.7700),
    "Kanakapura Road": (12.8718, 77.5457),
    "Thalaghattapura": (12.8710, 77.5350),
    "Anjanapura": (12.8620, 77.5620),
    "Kaggalipura": (12.8120, 77.5100),
    "Harohalli": (12.6700, 77.4600),

    # ── East Bengaluru & IT Corridor (ORR / Whitefield / Sarjapur)
    "Bellandur": (12.9250, 77.6766),
    "Green Glen Layout Bellandur": (12.9270, 77.6710),
    "Kadubeesanahalli": (12.9345, 77.6912),
    "Devarabeesanahalli": (12.9290, 77.6870),
    "Marathahalli": (12.9591, 77.6974),
    "Marathahalli ORR": (12.9550, 77.6990),
    "Munnekollal": (12.9510, 77.7120),
    "Kundalahalli": (12.9667, 77.7167),
    "Kundalahalli Gate": (12.9620, 77.7130),
    "Brookefield": (12.9654, 77.7185),
    "Whitefield": (12.9698, 77.7499),
    "ITPL Whitefield": (12.9856, 77.7315),
    "Hope Farm Whitefield": (12.9830, 77.7530),
    "Kadugodi": (12.9982, 77.7610),
    "Hoodi": (12.9918, 77.7161),
    "Hoodi Circle": (12.9890, 77.7140),
    "Mahadevapura": (12.9897, 77.6983),
    "Garudacharpalya": (12.9820, 77.7050),
    "Doddanekundi": (12.9720, 77.6980),
    "KR Puram": (13.0075, 77.6959),
    "Battarahalli": (13.0180, 77.7120),
    "Seegehalli": (13.0125, 77.7450),
    "Varthur": (12.9406, 77.7471),
    "Varthur Road": (12.9450, 77.7300),
    "Gunjur": (12.9280, 77.7380),
    "Panathur": (12.9348, 77.7051),
    "Balagere": (12.9380, 77.7180),
    "Sarjapur Road": (12.9105, 77.6850),
    "Sarjapur": (12.8601, 77.7865),
    "Haralur Road": (12.9022, 77.6603),
    "Harlur": (12.9100, 77.6650),
    "Kasavanahalli": (12.9080, 77.6713),
    "Kaikondrahalli": (12.9150, 77.6780),
    "Carmelaram": (12.9080, 77.7040),
    "Doddakannelli": (12.9130, 77.6920),
    "Somapura Sarjapur": (12.8750, 77.7600),
    "Dommasandra": (12.8820, 77.7550),
    "Chikka Tirupathi": (12.8900, 77.8600),
    "Old Airport Road": (12.9610, 77.6580),
    "Murugeshpalya": (12.9550, 77.6550),
    "Vimanapura": (12.9620, 77.6670),
    "HAL Layout": (12.9600, 77.6610),
    "CV Raman Nagar": (12.9855, 77.6639),
    "Kaggadasapura": (12.9839, 77.6811),
    "Malleshpalya": (12.9770, 77.6750),
    "Binnamangala": (12.9820, 77.6480),
    "Old Madras Road": (12.9980, 77.6700),
    "Budigere Cross": (13.0642, 77.7470),
    "Budigere": (13.0950, 77.7500),
    "Hoskote": (13.0700, 77.7980),

    # ── North Bengaluru & Airport Corridor
    "Hebbal": (13.0358, 77.5970),
    "Hebbal Kempapura": (13.0470, 77.6010),
    "Manyata Tech Park": (13.0483, 77.6209),
    "Nagawara": (13.0400, 77.6210),
    "Thanisandra": (13.0547, 77.6327),
    "Thanisandra Main Road": (13.0600, 77.6310),
    "Hennur": (13.0354, 77.6394),
    "Hennur Road": (13.0450, 77.6450),
    "Hennur Bande": (13.0550, 77.6480),
    "Horamavu": (13.0270, 77.6570),
    "Horamavu Agara": (13.0350, 77.6650),
    "Kalkere": (13.0380, 77.6740),
    "Ramamurthy Nagar": (13.0163, 77.6785),
    "Kalyan Nagar": (13.0221, 77.6403),
    "HRBR Layout": (13.0210, 77.6460),
    "HBR Layout": (13.0280, 77.6310),
    "Kammanahalli": (13.0094, 77.6375),
    "Banaswadi": (13.0070, 77.6510),
    "Kasturi Nagar": (13.0039, 77.6631),
    "Kothanur": (13.0569, 77.6517),
    "Geddalahalli": (13.0460, 77.6380),
    "Byrathi": (13.0670, 77.6580),
    "Bagalur": (13.1340, 77.6680),
    "Bagalur Main Road": (13.1250, 77.6550),
    "Jakkur": (13.0782, 77.6053),
    "Jakkur Aerodrome": (13.0750, 77.5990),
    "Amruthahalli": (13.0680, 77.5960),
    "Sahakara Nagar": (13.0623, 77.5878),
    "Kodigehalli": (13.0610, 77.5750),
    "Vidyaranyapura": (13.0784, 77.5574),
    "Thindlu": (13.0730, 77.5680),
    "Yelahanka": (13.1007, 77.5963),
    "Yelahanka New Town": (13.1050, 77.5820),
    "Yelahanka Air Force": (13.1350, 77.6080),
    "Kogilu": (13.1150, 77.6180),
    "Allalasandra": (13.0920, 77.5880),
    "Doddaballapur Road": (13.1500, 77.5700),
    "Doddaballapur": (13.2920, 77.5400),
    "Devanahalli": (13.2483, 77.7126),
    "Kempegowda Airport Area": (13.1986, 77.7066),
    "Chikkajala": (13.1650, 77.6350),

    # ── West & North-West Bengaluru
    "Malleshwaram": (13.0031, 77.5643),
    "Rajaji Nagar": (12.9982, 77.5530),
    "Rajajinagar 1st Block": (12.9920, 77.5580),
    "Basaveshwaranagar": (12.9873, 77.5385),
    "Mahalakshmi Layout": (13.0120, 77.5470),
    "Nandini Layout": (13.0180, 77.5390),
    "Yeshwanthpur": (13.0234, 77.5501),
    "Goraguntepalya": (13.0280, 77.5410),
    "Peenya": (13.0329, 77.5273),
    "Peenya Industrial Area": (13.0290, 77.5180),
    "Jalahalli": (13.0489, 77.5398),
    "Jalahalli West": (13.0520, 77.5250),
    "Jalahalli Cross": (13.0420, 77.5290),
    "Abbigere": (13.0720, 77.5280),
    "Mathikere": (13.0330, 77.5600),
    "Sanjay Nagar": (13.0380, 77.5740),
    "RMV 2nd Stage": (13.0320, 77.5760),
    "Vijayanagar": (12.9719, 77.5342),
    "Govindraj Nagar": (12.9690, 77.5310),
    "Nagarbhavi": (12.9567, 77.5097),
    "Nagarbhavi 1st Stage": (12.9620, 77.5150),
    "Nagarbhavi 2nd Stage": (12.9530, 77.5040),
    "Chandra Layout": (12.9580, 77.5250),
    "Attiguppe": (12.9610, 77.5290),
    "Nayandahalli": (12.9460, 77.5180),
    "Kengeri": (12.9081, 77.4839),
    "Kengeri Satellite Town": (12.9189, 77.4856),
    "Rajarajeshwari Nagar (RR Nagar)": (12.9260, 77.5200),
    "Mallasandra": (13.0500, 77.5050),
    "Thigalarapalya": (13.0180, 77.5020),
    "Nelamangala": (13.0980, 77.3910),
    "Bidadi": (12.7980, 77.3820),
    "Magadi Road": (12.9750, 77.5050),
    "Mysore Road": (12.9350, 77.5050),
}

# Default center of Bengaluru
BENGALURU_CENTER = (12.9716, 77.5946)

# ── Key Bengaluru Reference Points for Distance Engineering ──────────────────
BENGALURU_REFERENCE_POINTS = {
    "cbd": (12.9756, 77.6094),          # MG Road / Central Business District
    "itpl_whitefield": (12.9856, 77.7315),  # International Tech Park Bangalore (ITPL)
    "electronic_city": (12.8399, 77.6770),  # Electronic City IT Hub (Infosys, Wipro)
    "manyata_tech_park": (13.0483, 77.6209), # Manyata Embassy Business Park (North)
    "bellandur_ecospace": (12.9250, 77.6766), # Outer Ring Road Tech Corridor (RMZ Ecospace/Ecoworld)
    "airport": (13.1986, 77.7066),       # Kempegowda International Airport (BLR)
    "majestic_metro": (12.9781, 77.5696), # Majestic Metro & Railway Interchange
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two lat/lon coordinates."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * (2.0 * np.arcsin(np.sqrt(a)))


def get_locality_coordinates(locality_name: str) -> Tuple[float, float]:
    """
    Get (latitude, longitude) for a locality name, with fuzzy / fallback match.
    """
    if not locality_name:
        return BENGALURU_CENTER

    clean_name = locality_name.strip()

    # Exact match
    if clean_name in BENGALURU_LOCALITIES:
        return BENGALURU_LOCALITIES[clean_name]

    # Case-insensitive match
    for loc, coords in BENGALURU_LOCALITIES.items():
        if loc.lower() == clean_name.lower():
            return coords

    # Partial substring match
    for loc, coords in BENGALURU_LOCALITIES.items():
        if loc.lower() in clean_name.lower() or clean_name.lower() in loc.lower():
            return coords

    # Fallback to Bengaluru Center
    return BENGALURU_CENTER


def find_nearest_locality(lat: float, lon: float) -> Tuple[str, float]:
    """
    Find the closest known Bengaluru locality to given coordinates.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        Tuple of (nearest_locality_name, distance_in_km).
    """
    min_dist = float("inf")
    nearest_loc = "Bengaluru City"

    for loc, coords in BENGALURU_LOCALITIES.items():
        dist = _haversine_km(lat, lon, coords[0], coords[1])
        if dist < min_dist:
            min_dist = dist
            nearest_loc = loc

    return nearest_loc, min_dist
