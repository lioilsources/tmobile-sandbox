"""Sdílené konstanty dema 1."""
LLM = "http://127.0.0.1:8000/v1"
STT = "http://127.0.0.1:8001/v1/audio/transcriptions"
TTS = "http://127.0.0.1:8102/synthesize"
WHISPER_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
VOICE = {"operator": "piper-cs-kasandra-medium", "zakaznik": "piper-cs-jirka-medium"}
OPERATOR = "Telco Nova"   # fiktivní operátor
LABELS = {
    "kategorie": ["fakturace", "zmena_tarifu", "roaming", "technicka_zavada", "vypoved_smlouvy",
                  "prenos_cisla", "zarizeni_a_pojisteni", "sim_a_esim"],
    "sentiment_zakaznika": ["pozitivni", "neutralni", "negativni"],
    "dalsi_krok": ["zadny", "zpetne_volani", "technik_na_miste", "zaslani_dokumentu", "eskalace_na_specialistu", "dobropis"],
}
BOOLS = ["vyreseno", "riziko_odchodu", "overeni_identity"]

# Definice štítků — sdílí generátor (scenarios.py) i analýza (process.py), aby zadání a hodnocení mluvila stejným jazykem.
DEF_SENTIMENT = {"pozitivni": "zákazník na konci výslovně děkuje a je spokojený",
                 "neutralni": "zákazník hovor ukončí věcně, bez spokojenosti i bez stížnosti",
                 "negativni": "zákazník na konci vyjádří nespokojenost, zklamání nebo rozčilení"}
DEF_KROK = {"zadny": "nic dalšího se nedomlouvá – žádný e-mail, SMS, zpětné volání ani předání",
            "zpetne_volani": "operátorka slíbí, že zákazníkovi někdo zavolá zpět",
            "technik_na_miste": "objedná se návštěva technika u zákazníka a zákazník souhlasí",
            "zaslani_dokumentu": "zákazníkovi se pošle dokument, smlouva nebo formulář (e-mailem / poštou)",
            "eskalace_na_specialistu": "případ se předá specialistovi, reklamačnímu nebo technickému oddělení",
            "dobropis": "zákazník dostane dobropis nebo vrácení peněz"}
