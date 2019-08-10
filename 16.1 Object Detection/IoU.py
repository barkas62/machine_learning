def IoU(r1, r2):
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])
    dx = max(0, x2-x1+1)
    dy = max(0, y2-y1+1)
    I = dx*dy
    A1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
    A2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
    return I / (A1+A2-I)

r1 = (0,0, 9, 9)
r2 = (5,5, 14, 14)

r2 = (10,1,12,2) #0

res = IoU(r1,r2)
pass

