"""Serving-only scene/Pokemon protocol and loopback, uncached prompt diagnostics.

The archived EdgeBackend generation/preprocessing implementation is inherited
unchanged. This layer does not change its weights or scientific evaluation.
"""

import base64
import copy
import hashlib
import io
import ipaddress
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager

from PIL import Image

from brock_two.constants import PROMPT as LEGACY_PROMPT
from brock_two.edge import EdgeBackend

PROTOCOL = "brockone.scene-presence-identity.v2"
SCENE_PROMPT = (
    "Look at the image, not at suggested names. Describe the visible scene in at most 12 words. "
    "Identify a Pokemon only from its visually recognizable pictured body, toy, or character. "
    "Ordinary text, printed or screen species names, logos, and colors alone are not evidence of a Pokemon. "
    "Do not guess a species. If no Pokemon is visibly recognizable, default to no pokemon present. "
    "If a possible character is visible but not identifiable, use uncertain. "
    'Return only compact JSON: {"scene":"description","pokemon":{"status":"identified|absent|uncertain","name":null}}. '
    "For identified use its exact species name instead of null; otherwise name must be null."
)
PROMPTS = {
    "legacy": LEGACY_PROMPT,
    "scene_only": "Describe only the visible scene in one short sentence. Do not guess hidden objects. Readable names or text are not proof that an object is present.",
    "scene_pokemon": SCENE_PROMPT,
    "presence_only": 'Inspect only the pictured visual subject. Is a Pokemon body, toy, or character visibly recognizable? Text, species names, logos and colors alone do not establish presence. Do not guess a species. Return only JSON {"status":"identified|absent|uncertain","name":null}. Use absent and null if no Pokemon is visibly recognizable; uncertain and null if a possible character cannot be identified; identified and its exact species name only from visual appearance.',
    "visual_presence": "Is a non-human Pokemon creature visibly pictured in this image? Ignore all writing, names, logos and people. Answer exactly yes, no, or uncertain.",
    "visual_identity": "Identify a visibly pictured Pokemon creature from its body and shape, ignoring all text and logos. If no creature is visible, answer exactly: no pokemon present. If unsure, answer exactly: Pokemon not confirmed. Otherwise answer only its species name.",
    "scene_brief": "Describe the visible objects and setting in one short sentence. Do not read text or name characters.",
}
CANONICAL_NAMES = (
    "Bulbasaur",
    "Caterpie",
    "Voltorb",
    "Gholdengo",
    "Wo-Chien",
    "Chien-Pao",
    "Ting-Lu",
    "Chi-Yu",
    "Roaring Moon",
    "Iron Valiant",
    "Koraidon",
    "Miraidon",
    "Walking Wake",
    "Electrode",
    "Iron Leaves",
    "Dipplin",
    "Poltchageist",
    "Sinistcha",
    "Okidogi",
    "Munkidori",
    "Fezandipiti",
    "Ogerpon",
    "Archaludon",
    "Hydrapple",
    "Exeggcute",
    "Gouging Fire",
    "Raging Bolt",
    "Iron Boulder",
    "Iron Crown",
    "Terapagos",
    "Pecharunt",
    "Exeggutor",
    "Cubone",
    "Marowak",
    "Hitmonlee",
    "Hitmonchan",
    "Lickitung",
    "Koffing",
    "Metapod",
    "Weezing",
    "Rhyhorn",
    "Rhydon",
    "Chansey",
    "Tangela",
    "Kangaskhan",
    "Horsea",
    "Seadra",
    "Goldeen",
    "Seaking",
    "Butterfree",
    "Staryu",
    "Starmie",
    "Mr. Mime",
    "Scyther",
    "Jynx",
    "Electabuzz",
    "Magmar",
    "Pinsir",
    "Tauros",
    "Magikarp",
    "Weedle",
    "Gyarados",
    "Lapras",
    "Ditto",
    "Eevee",
    "Vaporeon",
    "Jolteon",
    "Flareon",
    "Porygon",
    "Omanyte",
    "Omastar",
    "Kakuna",
    "Kabuto",
    "Kabutops",
    "Aerodactyl",
    "Snorlax",
    "Articuno",
    "Zapdos",
    "Moltres",
    "Dratini",
    "Dragonair",
    "Dragonite",
    "Beedrill",
    "Mewtwo",
    "Mew",
    "Chikorita",
    "Bayleef",
    "Meganium",
    "Cyndaquil",
    "Quilava",
    "Typhlosion",
    "Totodile",
    "Croconaw",
    "Pidgey",
    "Feraligatr",
    "Sentret",
    "Furret",
    "Hoothoot",
    "Noctowl",
    "Ledyba",
    "Ledian",
    "Spinarak",
    "Ariados",
    "Crobat",
    "Pidgeotto",
    "Chinchou",
    "Lanturn",
    "Pichu",
    "Cleffa",
    "Igglybuff",
    "Togepi",
    "Togetic",
    "Natu",
    "Xatu",
    "Mareep",
    "Pidgeot",
    "Flaaffy",
    "Ampharos",
    "Bellossom",
    "Marill",
    "Azumarill",
    "Sudowoodo",
    "Politoed",
    "Hoppip",
    "Skiploom",
    "Jumpluff",
    "Rattata",
    "Aipom",
    "Sunkern",
    "Sunflora",
    "Yanma",
    "Wooper",
    "Quagsire",
    "Espeon",
    "Umbreon",
    "Murkrow",
    "Slowking",
    "Ivysaur",
    "Raticate",
    "Misdreavus",
    "Unown",
    "Wobbuffet",
    "Girafarig",
    "Pineco",
    "Forretress",
    "Dunsparce",
    "Gligar",
    "Steelix",
    "Snubbull",
    "Spearow",
    "Granbull",
    "Qwilfish",
    "Scizor",
    "Shuckle",
    "Heracross",
    "Sneasel",
    "Teddiursa",
    "Ursaring",
    "Slugma",
    "Magcargo",
    "Fearow",
    "Swinub",
    "Piloswine",
    "Corsola",
    "Remoraid",
    "Octillery",
    "Delibird",
    "Mantine",
    "Skarmory",
    "Houndour",
    "Houndoom",
    "Ekans",
    "Kingdra",
    "Phanpy",
    "Donphan",
    "Porygon2",
    "Stantler",
    "Smeargle",
    "Tyrogue",
    "Hitmontop",
    "Smoochum",
    "Elekid",
    "Arbok",
    "Magby",
    "Miltank",
    "Blissey",
    "Raikou",
    "Entei",
    "Suicune",
    "Larvitar",
    "Pupitar",
    "Tyranitar",
    "Lugia",
    "Pikachu",
    "Ho-Oh",
    "Celebi",
    "Treecko",
    "Grovyle",
    "Sceptile",
    "Torchic",
    "Combusken",
    "Blaziken",
    "Mudkip",
    "Marshtomp",
    "Raichu",
    "Swampert",
    "Poochyena",
    "Mightyena",
    "Zigzagoon",
    "Linoone",
    "Wurmple",
    "Silcoon",
    "Beautifly",
    "Cascoon",
    "Dustox",
    "Sandshrew",
    "Lotad",
    "Lombre",
    "Ludicolo",
    "Seedot",
    "Nuzleaf",
    "Shiftry",
    "Taillow",
    "Swellow",
    "Wingull",
    "Pelipper",
    "Sandslash",
    "Ralts",
    "Kirlia",
    "Gardevoir",
    "Surskit",
    "Masquerain",
    "Shroomish",
    "Breloom",
    "Slakoth",
    "Vigoroth",
    "Slaking",
    "Nidoran♀",
    "Nincada",
    "Ninjask",
    "Shedinja",
    "Whismur",
    "Loudred",
    "Exploud",
    "Makuhita",
    "Hariyama",
    "Azurill",
    "Nosepass",
    "Venusaur",
    "Nidorina",
    "Skitty",
    "Delcatty",
    "Sableye",
    "Mawile",
    "Aron",
    "Lairon",
    "Aggron",
    "Meditite",
    "Medicham",
    "Electrike",
    "Nidoqueen",
    "Manectric",
    "Plusle",
    "Minun",
    "Volbeat",
    "Illumise",
    "Roselia",
    "Gulpin",
    "Swalot",
    "Carvanha",
    "Sharpedo",
    "Nidoran♂",
    "Wailmer",
    "Wailord",
    "Numel",
    "Camerupt",
    "Torkoal",
    "Spoink",
    "Grumpig",
    "Spinda",
    "Trapinch",
    "Vibrava",
    "Nidorino",
    "Flygon",
    "Cacnea",
    "Cacturne",
    "Swablu",
    "Altaria",
    "Zangoose",
    "Seviper",
    "Lunatone",
    "Solrock",
    "Barboach",
    "Nidoking",
    "Whiscash",
    "Corphish",
    "Crawdaunt",
    "Baltoy",
    "Claydol",
    "Lileep",
    "Cradily",
    "Anorith",
    "Armaldo",
    "Feebas",
    "Clefairy",
    "Milotic",
    "Castform",
    "Kecleon",
    "Shuppet",
    "Banette",
    "Duskull",
    "Dusclops",
    "Tropius",
    "Chimecho",
    "Absol",
    "Clefable",
    "Wynaut",
    "Snorunt",
    "Glalie",
    "Spheal",
    "Sealeo",
    "Walrein",
    "Clamperl",
    "Huntail",
    "Gorebyss",
    "Relicanth",
    "Vulpix",
    "Luvdisc",
    "Bagon",
    "Shelgon",
    "Salamence",
    "Beldum",
    "Metang",
    "Metagross",
    "Regirock",
    "Regice",
    "Registeel",
    "Ninetales",
    "Latias",
    "Latios",
    "Kyogre",
    "Groudon",
    "Rayquaza",
    "Jirachi",
    "Deoxys",
    "Turtwig",
    "Grotle",
    "Torterra",
    "Jigglypuff",
    "Chimchar",
    "Monferno",
    "Infernape",
    "Piplup",
    "Prinplup",
    "Empoleon",
    "Starly",
    "Staravia",
    "Staraptor",
    "Bidoof",
    "Charmander",
    "Wigglytuff",
    "Bibarel",
    "Kricketot",
    "Kricketune",
    "Shinx",
    "Luxio",
    "Luxray",
    "Budew",
    "Roserade",
    "Cranidos",
    "Rampardos",
    "Zubat",
    "Shieldon",
    "Bastiodon",
    "Burmy",
    "Wormadam",
    "Mothim",
    "Combee",
    "Vespiquen",
    "Pachirisu",
    "Buizel",
    "Floatzel",
    "Golbat",
    "Cherubi",
    "Cherrim",
    "Shellos",
    "Gastrodon",
    "Ambipom",
    "Drifloon",
    "Drifblim",
    "Buneary",
    "Lopunny",
    "Mismagius",
    "Oddish",
    "Honchkrow",
    "Glameow",
    "Purugly",
    "Chingling",
    "Stunky",
    "Skuntank",
    "Bronzor",
    "Bronzong",
    "Bonsly",
    "Mime Jr.",
    "Gloom",
    "Happiny",
    "Chatot",
    "Spiritomb",
    "Gible",
    "Gabite",
    "Garchomp",
    "Munchlax",
    "Riolu",
    "Lucario",
    "Hippopotas",
    "Vileplume",
    "Hippowdon",
    "Skorupi",
    "Drapion",
    "Croagunk",
    "Toxicroak",
    "Carnivine",
    "Finneon",
    "Lumineon",
    "Mantyke",
    "Snover",
    "Paras",
    "Abomasnow",
    "Weavile",
    "Magnezone",
    "Lickilicky",
    "Rhyperior",
    "Tangrowth",
    "Electivire",
    "Magmortar",
    "Togekiss",
    "Yanmega",
    "Parasect",
    "Leafeon",
    "Glaceon",
    "Gliscor",
    "Mamoswine",
    "Porygon-Z",
    "Gallade",
    "Probopass",
    "Dusknoir",
    "Froslass",
    "Rotom",
    "Venonat",
    "Uxie",
    "Mesprit",
    "Azelf",
    "Dialga",
    "Palkia",
    "Heatran",
    "Regigigas",
    "Giratina",
    "Cresselia",
    "Phione",
    "Venomoth",
    "Manaphy",
    "Darkrai",
    "Shaymin",
    "Arceus",
    "Victini",
    "Snivy",
    "Servine",
    "Serperior",
    "Tepig",
    "Pignite",
    "Charmeleon",
    "Diglett",
    "Emboar",
    "Oshawott",
    "Dewott",
    "Samurott",
    "Patrat",
    "Watchog",
    "Lillipup",
    "Herdier",
    "Stoutland",
    "Purrloin",
    "Dugtrio",
    "Liepard",
    "Pansage",
    "Simisage",
    "Pansear",
    "Simisear",
    "Panpour",
    "Simipour",
    "Munna",
    "Musharna",
    "Pidove",
    "Meowth",
    "Tranquill",
    "Unfezant",
    "Blitzle",
    "Zebstrika",
    "Roggenrola",
    "Boldore",
    "Gigalith",
    "Woobat",
    "Swoobat",
    "Drilbur",
    "Persian",
    "Excadrill",
    "Audino",
    "Timburr",
    "Gurdurr",
    "Conkeldurr",
    "Tympole",
    "Palpitoad",
    "Seismitoad",
    "Throh",
    "Sawk",
    "Psyduck",
    "Sewaddle",
    "Swadloon",
    "Leavanny",
    "Venipede",
    "Whirlipede",
    "Scolipede",
    "Cottonee",
    "Whimsicott",
    "Petilil",
    "Lilligant",
    "Golduck",
    "Basculin",
    "Sandile",
    "Krokorok",
    "Krookodile",
    "Darumaka",
    "Darmanitan",
    "Maractus",
    "Dwebble",
    "Crustle",
    "Scraggy",
    "Mankey",
    "Scrafty",
    "Sigilyph",
    "Yamask",
    "Cofagrigus",
    "Tirtouga",
    "Carracosta",
    "Archen",
    "Archeops",
    "Trubbish",
    "Garbodor",
    "Primeape",
    "Zorua",
    "Zoroark",
    "Minccino",
    "Cinccino",
    "Gothita",
    "Gothorita",
    "Gothitelle",
    "Solosis",
    "Duosion",
    "Reuniclus",
    "Growlithe",
    "Ducklett",
    "Swanna",
    "Vanillite",
    "Vanillish",
    "Vanilluxe",
    "Deerling",
    "Sawsbuck",
    "Emolga",
    "Karrablast",
    "Escavalier",
    "Arcanine",
    "Foongus",
    "Amoonguss",
    "Frillish",
    "Jellicent",
    "Alomomola",
    "Joltik",
    "Galvantula",
    "Ferroseed",
    "Ferrothorn",
    "Klink",
    "Charizard",
    "Poliwag",
    "Klang",
    "Klinklang",
    "Tynamo",
    "Eelektrik",
    "Eelektross",
    "Elgyem",
    "Beheeyem",
    "Litwick",
    "Lampent",
    "Chandelure",
    "Poliwhirl",
    "Axew",
    "Fraxure",
    "Haxorus",
    "Cubchoo",
    "Beartic",
    "Cryogonal",
    "Shelmet",
    "Accelgor",
    "Stunfisk",
    "Mienfoo",
    "Poliwrath",
    "Mienshao",
    "Druddigon",
    "Golett",
    "Golurk",
    "Pawniard",
    "Bisharp",
    "Bouffalant",
    "Rufflet",
    "Braviary",
    "Vullaby",
    "Abra",
    "Mandibuzz",
    "Heatmor",
    "Durant",
    "Deino",
    "Zweilous",
    "Hydreigon",
    "Larvesta",
    "Volcarona",
    "Cobalion",
    "Terrakion",
    "Kadabra",
    "Virizion",
    "Tornadus",
    "Thundurus",
    "Reshiram",
    "Zekrom",
    "Landorus",
    "Kyurem",
    "Keldeo",
    "Meloetta",
    "Genesect",
    "Alakazam",
    "Chespin",
    "Quilladin",
    "Chesnaught",
    "Fennekin",
    "Braixen",
    "Delphox",
    "Froakie",
    "Frogadier",
    "Greninja",
    "Bunnelby",
    "Machop",
    "Diggersby",
    "Fletchling",
    "Fletchinder",
    "Talonflame",
    "Scatterbug",
    "Spewpa",
    "Vivillon",
    "Litleo",
    "Pyroar",
    "Flabébé",
    "Machoke",
    "Floette",
    "Florges",
    "Skiddo",
    "Gogoat",
    "Pancham",
    "Pangoro",
    "Furfrou",
    "Espurr",
    "Meowstic",
    "Honedge",
    "Machamp",
    "Doublade",
    "Aegislash",
    "Spritzee",
    "Aromatisse",
    "Swirlix",
    "Slurpuff",
    "Inkay",
    "Malamar",
    "Binacle",
    "Barbaracle",
    "Bellsprout",
    "Skrelp",
    "Dragalge",
    "Clauncher",
    "Clawitzer",
    "Helioptile",
    "Heliolisk",
    "Tyrunt",
    "Tyrantrum",
    "Amaura",
    "Aurorus",
    "Squirtle",
    "Weepinbell",
    "Sylveon",
    "Hawlucha",
    "Dedenne",
    "Carbink",
    "Goomy",
    "Sliggoo",
    "Goodra",
    "Klefki",
    "Phantump",
    "Trevenant",
    "Victreebel",
    "Pumpkaboo",
    "Gourgeist",
    "Bergmite",
    "Avalugg",
    "Noibat",
    "Noivern",
    "Xerneas",
    "Yveltal",
    "Zygarde",
    "Diancie",
    "Tentacool",
    "Hoopa",
    "Volcanion",
    "Rowlet",
    "Dartrix",
    "Decidueye",
    "Litten",
    "Torracat",
    "Incineroar",
    "Popplio",
    "Brionne",
    "Tentacruel",
    "Primarina",
    "Pikipek",
    "Trumbeak",
    "Toucannon",
    "Yungoos",
    "Gumshoos",
    "Grubbin",
    "Charjabug",
    "Vikavolt",
    "Crabrawler",
    "Geodude",
    "Crabominable",
    "Oricorio",
    "Cutiefly",
    "Ribombee",
    "Rockruff",
    "Lycanroc",
    "Wishiwashi",
    "Mareanie",
    "Toxapex",
    "Mudbray",
    "Graveler",
    "Mudsdale",
    "Dewpider",
    "Araquanid",
    "Fomantis",
    "Lurantis",
    "Morelull",
    "Shiinotic",
    "Salandit",
    "Salazzle",
    "Stufful",
    "Golem",
    "Bewear",
    "Bounsweet",
    "Steenee",
    "Tsareena",
    "Comfey",
    "Oranguru",
    "Passimian",
    "Wimpod",
    "Golisopod",
    "Sandygast",
    "Ponyta",
    "Palossand",
    "Pyukumuku",
    "Type: Null",
    "Silvally",
    "Minior",
    "Komala",
    "Turtonator",
    "Togedemaru",
    "Mimikyu",
    "Bruxish",
    "Rapidash",
    "Drampa",
    "Dhelmise",
    "Jangmo-o",
    "Hakamo-o",
    "Kommo-o",
    "Tapu Koko",
    "Tapu Lele",
    "Tapu Bulu",
    "Tapu Fini",
    "Cosmog",
    "Slowpoke",
    "Cosmoem",
    "Solgaleo",
    "Lunala",
    "Nihilego",
    "Buzzwole",
    "Pheromosa",
    "Xurkitree",
    "Celesteela",
    "Kartana",
    "Guzzlord",
    "Wartortle",
    "Slowbro",
    "Necrozma",
    "Magearna",
    "Marshadow",
    "Poipole",
    "Naganadel",
    "Stakataka",
    "Blacephalon",
    "Zeraora",
    "Meltan",
    "Melmetal",
    "Magnemite",
    "Grookey",
    "Thwackey",
    "Rillaboom",
    "Scorbunny",
    "Raboot",
    "Cinderace",
    "Sobble",
    "Drizzile",
    "Inteleon",
    "Skwovet",
    "Magneton",
    "Greedent",
    "Rookidee",
    "Corvisquire",
    "Corviknight",
    "Blipbug",
    "Dottler",
    "Orbeetle",
    "Nickit",
    "Thievul",
    "Gossifleur",
    "Farfetch’d",
    "Eldegoss",
    "Wooloo",
    "Dubwool",
    "Chewtle",
    "Drednaw",
    "Yamper",
    "Boltund",
    "Rolycoly",
    "Carkol",
    "Coalossal",
    "Doduo",
    "Applin",
    "Flapple",
    "Appletun",
    "Silicobra",
    "Sandaconda",
    "Cramorant",
    "Arrokuda",
    "Barraskewda",
    "Toxel",
    "Toxtricity",
    "Dodrio",
    "Sizzlipede",
    "Centiskorch",
    "Clobbopus",
    "Grapploct",
    "Sinistea",
    "Polteageist",
    "Hatenna",
    "Hattrem",
    "Hatterene",
    "Impidimp",
    "Seel",
    "Morgrem",
    "Grimmsnarl",
    "Obstagoon",
    "Perrserker",
    "Cursola",
    "Sirfetch’d",
    "Mr. Rime",
    "Runerigus",
    "Milcery",
    "Alcremie",
    "Dewgong",
    "Falinks",
    "Pincurchin",
    "Snom",
    "Frosmoth",
    "Stonjourner",
    "Eiscue",
    "Indeedee",
    "Morpeko",
    "Cufant",
    "Copperajah",
    "Grimer",
    "Dracozolt",
    "Arctozolt",
    "Dracovish",
    "Arctovish",
    "Duraludon",
    "Dreepy",
    "Drakloak",
    "Dragapult",
    "Zacian",
    "Zamazenta",
    "Muk",
    "Eternatus",
    "Kubfu",
    "Urshifu",
    "Zarude",
    "Regieleki",
    "Regidrago",
    "Glastrier",
    "Spectrier",
    "Calyrex",
    "Wyrdeer",
    "Blastoise",
    "Shellder",
    "Kleavor",
    "Ursaluna",
    "Basculegion",
    "Sneasler",
    "Overqwil",
    "Enamorus",
    "Sprigatito",
    "Floragato",
    "Meowscarada",
    "Fuecoco",
    "Cloyster",
    "Crocalor",
    "Skeledirge",
    "Quaxly",
    "Quaxwell",
    "Quaquaval",
    "Lechonk",
    "Oinkologne",
    "Tarountula",
    "Spidops",
    "Nymble",
    "Gastly",
    "Lokix",
    "Pawmi",
    "Pawmo",
    "Pawmot",
    "Tandemaus",
    "Maushold",
    "Fidough",
    "Dachsbun",
    "Smoliv",
    "Dolliv",
    "Haunter",
    "Arboliva",
    "Squawkabilly",
    "Nacli",
    "Naclstack",
    "Garganacl",
    "Charcadet",
    "Armarouge",
    "Ceruledge",
    "Tadbulb",
    "Bellibolt",
    "Gengar",
    "Wattrel",
    "Kilowattrel",
    "Maschiff",
    "Mabosstiff",
    "Shroodle",
    "Grafaiai",
    "Bramblin",
    "Brambleghast",
    "Toedscool",
    "Toedscruel",
    "Onix",
    "Klawf",
    "Capsakid",
    "Scovillain",
    "Rellor",
    "Rabsca",
    "Flittle",
    "Espathra",
    "Tinkatink",
    "Tinkatuff",
    "Tinkaton",
    "Drowzee",
    "Wiglett",
    "Wugtrio",
    "Bombirdier",
    "Finizen",
    "Palafin",
    "Varoom",
    "Revavroom",
    "Cyclizar",
    "Orthworm",
    "Glimmet",
    "Hypno",
    "Glimmora",
    "Greavard",
    "Houndstone",
    "Flamigo",
    "Cetoddle",
    "Cetitan",
    "Veluza",
    "Dondozo",
    "Tatsugiri",
    "Annihilape",
    "Krabby",
    "Clodsire",
    "Farigiraf",
    "Dudunsparce",
    "Kingambit",
    "Great Tusk",
    "Scream Tail",
    "Brute Bonnet",
    "Flutter Mane",
    "Slither Wing",
    "Sandy Shocks",
    "Kingler",
    "Iron Treads",
    "Iron Bundle",
    "Iron Hands",
    "Iron Jugulis",
    "Iron Moth",
    "Iron Thorns",
    "Frigibax",
    "Arctibax",
    "Baxcalibur",
    "Gimmighoul",
)
CANONICAL = {name.casefold(): name for name in CANONICAL_NAMES}


def exact_image_identity(image):
    rgb = image.convert("RGB")
    descriptor = {"mode": "RGB", "width": rgb.width, "height": rgb.height}
    h = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode() + b"\0")
    h.update(rgb.tobytes())
    return {**descriptor, "sha256": h.hexdigest(), "scope": "original_decoded_RGB_and_geometry_before_resize"}


def parse_model_text(raw):
    result = {
        "scene": None,
        "pokemon": {"status": "uncertain", "name": None},
        "pokemon_caption": "Pokémon not confirmed",
        "raw_model_text": raw,
        "parse_error": None,
        "recognition_verified": False,
    }
    try:

        def pairs(items):
            out = {}
            for key, value in items:
                if key in out:
                    raise ValueError("duplicate JSON key")
                out[key] = value
            return out

        value = json.loads(raw, object_pairs_hook=pairs)
        if not isinstance(value, dict) or set(value) != {"scene", "pokemon"}:
            raise ValueError("exact scene and pokemon object required")
        scene = value["scene"]
        if (
            not isinstance(scene, str)
            or not scene.strip()
            or len(scene) > 240
            or any(ord(c) < 32 for c in scene)
        ):
            raise ValueError("short plain scene required")
        pokemon = value["pokemon"]
        if not isinstance(pokemon, dict) or set(pokemon) != {"status", "name"}:
            raise ValueError("exact status/name required")
        status, name = pokemon["status"], pokemon["name"]
        if status not in ("identified", "absent", "uncertain"):
            raise ValueError("unknown status")
        if status == "identified":
            if not isinstance(name, str) or name.strip().casefold() not in CANONICAL:
                raise ValueError("canonical species name required")
            name = CANONICAL[name.strip().casefold()]
        elif name is not None:
            raise ValueError("abstention cannot carry a species")
        result.update(scene=scene.strip(), pokemon={"status": status, "name": name})
        result["pokemon_caption"] = (
            f"This is {name}."
            if status == "identified"
            else ("no pokemon present" if status == "absent" else "Pokémon not confirmed")
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        result["parse_error"] = str(exc)
    return result


def strict_presence(raw):
    if not isinstance(raw, str):
        return "uncertain"
    match = re.fullmatch(r"(yes|no|uncertain)[.!?]?", raw.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else "uncertain"


def strict_identity(raw):
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.casefold() in CANONICAL:
        return CANONICAL[text.casefold()]
    if text[-1:] in (".", "!", "?"):
        return CANONICAL.get(text[:-1].casefold())
    return None


def short_policy(scene_raw, presence_raw, identity_raw=None):
    presence = strict_presence(presence_raw)
    name = strict_identity(identity_raw) if presence == "yes" else None
    status = "absent" if presence == "no" else ("identified" if name else "uncertain")
    scene = " ".join(scene_raw.split()) if isinstance(scene_raw, str) else ""
    scene_valid = bool(scene) and len(scene) <= 500 and not any(ord(c) < 32 for c in scene)
    return {
        "scene": scene if scene_valid else "Scene not confirmed.",
        "pokemon": {"status": status, "name": name},
        "pokemon_caption": "no pokemon present"
        if status == "absent"
        else (f"This is {name}." if name else "Pokémon not confirmed"),
        "presence_decision": presence,
        "parse_error": None if status != "uncertain" else "presence_or_identity_not_confirmed",
        "recognition_verified": False,
        "scene_verified": False,
    }


class PromptedEdgeBackend(EdgeBackend):
    def __init__(self, *args, **kwargs):
        self._prompt_lock = threading.RLock()
        self._serving_prompt = LEGACY_PROMPT
        super().__init__(*args, **kwargs)
        self.metadata.update(
            serving_protocol=PROTOCOL,
            live_policy_version=PROTOCOL,
            public_request_aliases=[LEGACY_PROMPT, SCENE_PROMPT],
            actual_policy_prompts={
                k: PROMPTS[k] for k in ("scene_brief", "visual_presence", "visual_identity")
            },
            prompt_diagnostic_variants=list(PROMPTS),
            response_cache_enabled=False,
            recognition_validation="unvalidated_live_trial",
        )

    def _request(self, image_buffer):
        # Same native request fields as the archived method, with explicit text.
        contents = [
            self.rt.MessageContent("image", "inline"),
            self.rt.MessageContent("text", self._serving_prompt),
        ]
        inner = self.rt.Request([self.rt.Message("user", contents)])
        if image_buffer is not None:
            inner.image_buffers = [image_buffer]
        request = self.rt.LLMGenerationRequest()
        request.requests = [inner]
        request.max_generate_length = self.max_new_tokens
        request.temperature, request.top_p, request.top_k = 0.0, 1.0, 1
        request.apply_chat_template, request.add_generation_prompt = True, True
        if not hasattr(request, "enable_thinking"):
            raise RuntimeError("Native binding cannot explicitly disable thinking")
        request.enable_thinking = False
        request.num_logprobs = 1
        request.save_system_prompt_kv_cache = False
        if hasattr(self.rt, "ContextCacheLookupPolicy"):
            request.context_cache_lookup_policy = self.rt.ContextCacheLookupPolicy.BYPASS
        return request

    def predict_prompt(self, image, variant):
        if variant not in PROMPTS:
            raise ValueError("Unknown diagnostic prompt variant")
        with self._prompt_lock:
            old_prompt, old_count = self._serving_prompt, self._placeholder_prompt_tokens
            try:
                self._serving_prompt = PROMPTS[variant]
                counts = self.runtime.count_prompt_tokens(self._request(None))
                if len(counts) != 1 or type(counts[0]) is not int or counts[0] < 1:
                    raise RuntimeError("Native prompt count failed")
                self._placeholder_prompt_tokens = counts[0]
                result = super().predict(image)
                return {
                    **result,
                    "prompt_variant": variant,
                    "prompt": self._serving_prompt,
                    "prompt_sha256": hashlib.sha256(self._serving_prompt.encode()).hexdigest(),
                    "placeholder_prompt_tokens": counts[0],
                }
            finally:
                self._serving_prompt, self._placeholder_prompt_tokens = old_prompt, old_count


class SceneService:
    def __init__(self, backend, *, cache_enabled=False, capacity=16):
        if type(cache_enabled) is not bool or type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("Bounded explicit response cache required")
        self.backend, self.cache_enabled, self.capacity = backend, cache_enabled, capacity
        self._lock, self._cache = threading.Lock(), OrderedDict()

    def infer(self, image, variant, *, diagnostic=False):
        if variant not in PROMPTS:
            raise ValueError("Unknown prompt variant")
        rgb = image.convert("RGB")
        identity = exact_image_identity(rgb)
        prompt_sha = hashlib.sha256(PROMPTS[variant].encode()).hexdigest()
        # Bound to this one backend instance; no cache survives model reload.
        key = (
            identity["sha256"],
            prompt_sha,
            PROTOCOL,
            self.backend.image_tokens,
            self.backend.max_new_tokens,
        )
        with self._lock:
            if self.cache_enabled and not diagnostic and key in self._cache:
                original = self._cache.pop(key)
                self._cache[key] = original
                cached = copy.deepcopy(original)
                cached["brockone"].update(
                    cache_hit=True,
                    model_executed=False,
                    response_cache_age_ms=(time.monotonic() - cached.pop("_cached_at")) * 1000,
                )
                return cached
            trace = self.backend.predict_prompt(rgb, variant)
            raw = trace["text"]
            semantic = (
                parse_model_text(raw)
                if variant == "scene_pokemon"
                else {
                    "scene": raw if variant == "scene_only" else None,
                    "pokemon": {"status": "uncertain", "name": None},
                    "pokemon_caption": "Pokémon not confirmed",
                    "raw_model_text": raw,
                    "parse_error": "non_structured_diagnostic_or_legacy_prompt",
                    "recognition_verified": False,
                }
            )
            evidence_id = "chatcmpl-" + uuid.uuid4().hex
            block = {
                **semantic,
                "protocol": PROTOCOL,
                "evidence_id": evidence_id,
                "input_identity": identity["sha256"],
                "input_identity_details": identity,
                "cache_hit": False,
                "response_cache_enabled": self.cache_enabled,
                "model_executed": True,
                "diagnostic": diagnostic,
                "prompt_variant": variant,
                "prompt_sha256": prompt_sha,
                "native_trace": trace,
                "native_image_cache_enabled": self.backend.metadata.get("image_cache_enabled"),
                "native_image_cache_bypass": self.backend.metadata.get("image_cache_bypass"),
                "context_cache_enabled": self.backend.metadata.get("context_cache_enabled"),
            }
            caption = (
                raw
                if variant != "scene_pokemon"
                else ((semantic["scene"] + " " if semantic["scene"] else "") + semantic["pokemon_caption"])
            )
            result = {
                "id": evidence_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "brockone",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": caption},
                        "finish_reason": trace.get("finish_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": trace.get("input_tokens"),
                    "completion_tokens": trace.get("generated_tokens"),
                },
                "brock_two": {k: v for k, v in trace.items() if k != "text"},
                "brockone": block,
            }
            if self.cache_enabled and not diagnostic:
                self._cache[key] = {**copy.deepcopy(result), "_cached_at": time.monotonic()}
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
            return result

    def infer_policy(self, image):
        # Every public request executes the policy afresh. No temporal or response cache.
        started = time.monotonic()
        rgb = image.convert("RGB")
        identity = exact_image_identity(rgb)
        traces = []
        with self._lock:

            def stage(variant):
                trace = self.backend.predict_prompt(rgb, variant)
                trace = {
                    **trace,
                    "prompt_variant": variant,
                    "prompt": PROMPTS[variant],
                    "prompt_sha256": hashlib.sha256(PROMPTS[variant].encode()).hexdigest(),
                }
                if traces:
                    for key in ("prepared_image_sha256", "prepared_image_size", "actual_image_tokens"):
                        if trace.get(key) != traces[0].get(key):
                            raise RuntimeError(
                                "Policy stages did not use identical prepared image geometry/pixels"
                            )
                traces.append(trace)
                return trace["text"]

            scene_raw = stage("scene_brief")
            presence_raw = stage("visual_presence")
            identity_raw = stage("visual_identity") if strict_presence(presence_raw) == "yes" else None
            if exact_image_identity(rgb) != identity:
                raise RuntimeError("Original RGB image changed during policy stages")
            semantic = short_policy(scene_raw, presence_raw, identity_raw)
            extrema = rgb.getextrema()
            uniform = all(low == high for low, high in extrema)
            pixel_rule = {
                "rule": "exact_original_RGB_uniform_channels",
                "matched": uniform,
                "scope": "scene_wording_only_after_model_presence_no",
                "applied": False,
            }
            if uniform and semantic["presence_decision"] == "no":
                semantic["scene"] = "A uniform-color image with no visible objects."
                semantic["scene_verified"] = True
                pixel_rule.update(applied=True, rgb=[low for low, _ in extrema])
            evidence_id = "chatcmpl-" + uuid.uuid4().hex
            caption = f"Scene: {semantic['scene']}\nPokémon: {semantic['pokemon_caption']}"
            block = {
                **semantic,
                "protocol": PROTOCOL,
                "evidence_id": evidence_id,
                "input_identity": identity["sha256"],
                "input_identity_details": identity,
                "cache_hit": False,
                "response_cache_enabled": False,
                "model_executed": True,
                "diagnostic": False,
                "prompt_variant": "scene_presence_identity",
                "raw_model_text": json.dumps(
                    {"scene": scene_raw, "presence": presence_raw, "identity": identity_raw},
                    ensure_ascii=False,
                ),
                "policy_wall_ms": (time.monotonic() - started) * 1000,
                "native_trace": traces[-1],
                "stages": traces,
                "pixel_rule": pixel_rule,
                "native_image_cache_enabled": self.backend.metadata.get("image_cache_enabled"),
                "native_image_cache_bypass": self.backend.metadata.get("image_cache_bypass"),
                "context_cache_enabled": self.backend.metadata.get("context_cache_enabled"),
            }
            return {
                "id": evidence_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "brockone",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": caption},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": sum(t.get("input_tokens", 0) for t in traces),
                    "completion_tokens": sum(t.get("generated_tokens", 0) for t in traces),
                },
                "brock_two": {
                    **{k: v for k, v in traces[-1].items() if k != "text"},
                    "scope": "last_policy_stage",
                    "policy_stage_count": len(traces),
                },
                "brockone": block,
            }


def decode_image(url):
    if not isinstance(url, str) or not url.startswith(
        ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")
    ):
        raise ValueError("Inline PNG/JPEG/WebP required")
    if len(url) > 24 * 1024 * 1024:
        raise ValueError("Image request too large")
    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("Image pixel limit exceeded")
        return image.convert("RGB")


def create_app(backend):
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel, ConfigDict, Field

    class CompletionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model: str = "brockone"
        messages: list[dict]
        max_tokens: int = Field(default=64, ge=1, le=64)
        temperature: float = 0
        stream: bool = False
        image_tokens: int = 512

    class DiagnosticRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        variant: str
        image_url: str

    service = SceneService(backend, cache_enabled=getattr(backend, "response_cache_enabled", False))
    app = FastAPI(title="brockone", version="scene-presence-identity-2")
    admission = threading.Lock()

    @contextmanager
    def admit():
        # Admission precedes image decode, but FastAPI has already parsed the body.
        if not admission.acquire(blocking=False):
            raise HTTPException(429, "Inference busy", headers={"Retry-After": "1"})
        try:
            yield
        finally:
            admission.release()

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "configuration": backend.metadata,
            "serving_protocol": PROTOCOL,
            "response_cache_enabled": False,
            "live_policy_version": PROTOCOL,
            "public_policy_stage_cap": 3,
            "output_token_cap_per_stage": backend.max_new_tokens,
            "recognition_validated": False,
        }

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "brockone", "object": "model", "owned_by": "nv-asotelo"}]}

    @app.post("/internal/diagnostic")
    def diagnostic(body: DiagnosticRequest, request: Request):
        try:
            local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
        except ValueError:
            local = False
        if not local:
            raise HTTPException(403, "Loopback diagnostics only")
        if body.variant not in (
            "scene_only",
            "scene_pokemon",
            "presence_only",
            "visual_presence",
            "visual_identity",
            "scene_brief",
        ):
            raise HTTPException(400, "Unknown diagnostic variant")
        with admit():
            try:
                image = decode_image(body.image_url)
            except Exception as exc:
                raise HTTPException(400, "Invalid diagnostic image") from exc
            try:
                return service.infer(image, body.variant, diagnostic=True)
            finally:
                image.close()

    @app.post("/v1/chat/completions")
    def completion(request: CompletionRequest):
        if request.stream or request.temperature != 0 or request.model != "brockone":
            raise HTTPException(400, "brockone non-streaming greedy request required")
        if request.image_tokens != backend.image_tokens or request.max_tokens != backend.max_new_tokens:
            raise HTTPException(400, "Request budgets must match resident configuration")
        images, texts = [], []
        for message in request.messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                raise HTTPException(400, "Exactly one user image and fixed instruction required")
            for item in message["content"]:
                if not isinstance(item, dict):
                    raise HTTPException(400, "Malformed content")
                if item.get("type") == "image_url" and isinstance(item.get("image_url"), dict):
                    images.append(item["image_url"].get("url"))
                elif item.get("type") == "text":
                    texts.append(item.get("text"))
                else:
                    raise HTTPException(400, "Unsupported content")
        if (
            len(request.messages) != 1
            or len(images) != 1
            or len(texts) != 1
            or texts[0] not in (LEGACY_PROMPT, SCENE_PROMPT)
        ):
            raise HTTPException(400, "Only reviewed legacy or scene instruction accepted")
        with admit():
            try:
                image = decode_image(images[0])
            except Exception as exc:
                raise HTTPException(400, "Invalid image") from exc
            try:
                return service.infer_policy(image)
            finally:
                image.close()

    return app


def backend_from_args(args):
    backend = PromptedEdgeBackend(
        args.edge_engine_dir,
        args.edge_onnx_llm_dir,
        bindings_dir=args.edge_bindings_dir,
        plugin=args.edge_plugin,
        image_tokens=args.image_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    backend.response_cache_enabled = False  # No response caching until raw stability is measured.
    return backend
