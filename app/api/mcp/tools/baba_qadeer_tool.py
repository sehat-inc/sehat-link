import random

BABA_QADEER_LINES = [
    "Beta, chai thandi ho jaye toh zindagi garam nahin hoti.",
    "Jo banda subah uth kar bistar theek kare, woh aadhi jang jeet leta hai.",
    "Agar plan A fail ho jaye, alphabet mein aur 25 letters hotay hain.",
    "Zindagi aik dafa milti hai, magar mistakes repeat hoti rehti hain.",
    "Aqalmand woh hai jo galti se seekhe, aur philosopher woh jo baar baar kare.",
    "Beta, free advice ka koi mol nahi… is liye yeh bilkul free hai.",
    "Aaj ka tension kal ka ulcer ban sakta hai.",
    "Jis din tum haste ho, us din duniya thodi behtar lagti hai.",
    "Zindagi aik cricket match hai—kabhi googly, kabhi yorker.",
    "Jo cheez mil jaye, usko appreciate karo. Jo na mile, usko chor do."
]

def ask_baba_qadeer():
    """
    Returns a single random line of Baba Qadeer's wisdom.
    """
    return random.choice(BABA_QADEER_LINES)
