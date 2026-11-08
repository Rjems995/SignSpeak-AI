"""ASL alphabet lessons; J and Z need movement rather than a still handshape."""
LETTERS = {
    "A": {"title": "A little fist, a big beginning.", "steps": ["Make a gentle fist.", "Rest your thumb alongside the index finger, pointing up.", "Keep your palm facing forward and your thumb visible."], "group": "Closed hand"},
    "B": {"title": "Stand tall. Keep it together.", "steps": ["Hold four fingers straight up and together.", "Fold your thumb across your palm.", "Face your palm forward."], "group": "Open hand"},
    "C": {"title": "Give your hand a gentle curve.", "steps": ["Curve all four fingers as if holding a small cup.", "Curve your thumb opposite your fingers into a C.", "Turn slightly so the opening is visible."], "group": "Curved hand"},
    "D": {"title": "One finger points the way.", "steps": ["Point your index finger upward.", "Touch your other fingertips toward your thumb to form a circle.", "Keep the index finger extended above that circle."], "group": "One finger"},
    "E": {"title": "A small, relaxed fold.", "steps": ["Bend your four fingers down toward your palm.", "Bring the fingertips toward your folded thumb.", "Keep your fingers together and palm facing forward."], "group": "Closed hand"},
    "F": {"title": "A circle and three fingers.", "steps": ["Touch your index fingertip to your thumb tip.", "Extend your middle, ring, and little fingers upward.", "Keep the small circle and the three raised fingers visible."], "group": "Circle hand"},
    "G": {"title": "Point to the side.", "steps": ["Extend your index finger sideways.", "Hold your thumb roughly parallel with a small gap.", "Curl the other fingers and show the side of your hand."], "group": "Sideways hand"},
    "H": {"title": "Two fingers, side by side.", "steps": ["Extend your index and middle fingers together.", "Point them sideways with your hand turned to the side.", "Fold your ring and little fingers beneath your thumb."], "group": "Sideways hand"},
    "I": {"title": "Let your little finger stand tall.", "steps": ["Make a fist and extend only the little finger upward.", "Keep your thumb across the folded fingers.", "Hold still; tracing a hook with this handshape forms J."], "group": "One finger"},
    "J": {"title": "Draw a little J in the air.", "steps": ["Begin with the I handshape, little finger extended.", "Move your little fingertip down and around a hook to trace J.", "Select Record J / Z, trace once, and wait for the movement result."], "group": "Movement", "motion": True},
    "K": {"title": "Two fingers and a supporting thumb.", "steps": ["Extend and separate your index and middle fingers upward.", "Place your thumb against the base/side of the middle finger between the two fingers.", "Curl your ring and little fingers into your palm."], "group": "Two fingers"},
    "L": {"title": "Make room for an L.", "steps": ["Point your index finger upward.", "Extend your thumb sideways to make an L.", "Curl the other three fingers into your palm."], "group": "Two fingers"},
    "M": {"title": "Three fingers over your thumb.", "steps": ["Fold your thumb into your palm.", "Fold the index, middle, and ring fingers over it.", "Keep the little finger folded, with the thumb beneath three fingers."], "group": "Closed hand"},
    "N": {"title": "Two fingers over your thumb.", "steps": ["Fold your thumb into your palm.", "Fold your index and middle fingers over the thumb.", "Keep your ring and little fingers folded; the thumb sits beneath two fingers."], "group": "Closed hand"},
    "O": {"title": "Bring your fingertips together.", "steps": ["Curve your four fingers toward the thumb.", "Touch the fingertips to your thumb to form an O.", "Turn slightly so the round opening is visible."], "group": "Circle hand"},
    "P": {"title": "K takes a downward turn.", "steps": ["Begin with K: index and middle fingers extended, thumb against the middle finger.", "Turn downward so the middle finger points toward the floor.", "Keep your ring and little fingers folded."], "group": "Downward hand"},
    "Q": {"title": "G points toward the ground.", "steps": ["Begin with G: index finger and thumb extended with a gap.", "Turn your hand so those digits point downward.", "Curl the other fingers and keep the downward direction visible."], "group": "Downward hand"},
    "R": {"title": "Cross your fingers.", "steps": ["Extend your index and middle fingers upward.", "Cross your middle finger over the index finger.", "Hold the folded ring and little fingers down with your thumb."], "group": "Two fingers"},
    "S": {"title": "A fist, with the thumb across.", "steps": ["Make a fist with all four fingers folded.", "Place your thumb across the front of the fingers.", "Keep the thumb across the fist rather than alongside it as in A."], "group": "Closed hand"},
    "T": {"title": "A thumb between two fingers.", "steps": ["Make a relaxed fist.", "Tuck the thumb between the index and middle fingers.", "Let the thumb tip show between them."], "group": "Closed hand"},
    "U": {"title": "Two fingers, close together.", "steps": ["Extend the index and middle fingers upward.", "Keep them straight and touching without crossing.", "Hold the other fingers folded with your thumb."], "group": "Two fingers"},
    "V": {"title": "Two fingers. A little space.", "steps": ["Extend the index and middle fingers upward.", "Separate them into a V with your palm forward.", "Hold the remaining fingers folded with your thumb."], "group": "Two fingers"},
    "W": {"title": "Three fingers make a W.", "steps": ["Extend the index, middle, and ring fingers upward.", "Separate those three fingers slightly.", "Fold your little finger down and hold it with your thumb."], "group": "Three fingers"},
    "X": {"title": "Give your index finger a hook.", "steps": ["Make a fist and raise your index finger.", "Bend the index finger into a hook.", "Keep your other fingers folded and the hook visible."], "group": "Hooked hand"},
    "Y": {"title": "A friendly stretch.", "steps": ["Extend your thumb and little finger outward.", "Fold your index, middle, and ring fingers into your palm.", "Keep both extended digits visible."], "group": "Two fingers"},
    "Z": {"title": "Trace the shape of a Z.", "steps": ["Extend your index finger and fold the other fingers into your palm.", "Trace Z in the air: across, diagonally back, then across.", "Select Record J / Z and complete the movement once."], "group": "Movement", "motion": True},
}
STATIC_LETTERS = tuple(letter for letter, info in LETTERS.items() if not info.get("motion"))
MOTION_LETTERS = ("J", "Z")
DEFAULT_SETTINGS = {"theme": "light", "threshold": 0.85, "hold_ms": 900, "landmarks": True,
                    "save_history": True, "rate": 1.0, "voice": "", "camera": ""}

