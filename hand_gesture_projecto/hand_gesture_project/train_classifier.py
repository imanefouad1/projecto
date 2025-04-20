
def classify_gesture(landmarks):
    if landmarks is None:
        return None

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_ip = landmarks[3]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    def is_extended(tip, mcp): return tip.y < mcp.y

    thumb_extended = thumb_tip.x < thumb_ip.x
    index_extended = is_extended(index_tip, index_mcp)
    middle_extended = is_extended(middle_tip, middle_mcp)
    ring_extended = is_extended(ring_tip, ring_mcp)
    pinky_extended = is_extended(pinky_tip, pinky_mcp)

    if index_extended and pinky_extended and not middle_extended and not ring_extended and thumb_extended:
        return "I Love You"
    elif index_extended and thumb_extended and middle_extended and ring_extended and pinky_extended:
        return "Open Palm"
    elif index_extended and middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
        return "Peace"
    elif index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
        return "Pointer"
    elif thumb_tip.x < index_tip.x < middle_tip.x and not ring_extended and not pinky_extended:
        return "Okay"

    return "Unknown"
