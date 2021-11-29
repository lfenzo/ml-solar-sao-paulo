"""
Implementação do modelo empírico em python

l = [ (a + b*cos(W) * r_0) / f_c  ] * H

"""
from numpy import cos, sin, arccos, tan, radians, pi, rad2deg

class CPRGModel:

    def predict(self, lat, lon, LT, Ls, doy, H, correction = 0.15):

        B = (360 * (doy - 81)) / 365
        ET = (9.87 * sin(radians(2*B))) - (7.53 * cos(radians(B))) - (1.5 * cos(radians(B)))

        ts = LT + (ET/60) + (4/60)*(Ls - lon) # deve ser em graus

        G = (2 * pi * (doy - 1)) / 365 # radianos
        gamma = (180 / pi) * ( 0.006918          - 0.399912*cos(G)   + \
                               0.070257*sin(G)   - 0.006758*cos(2*G) + \
                               0.000907*sin(2*G) - 0.002697*cos(3*G) + \
                               0.001480*sin(3*G) ) # tem que ser convertido em radianos

        Ws = rad2deg( arccos( -tan(radians(lat)) * tan(radians(gamma)) ) )
        W = (360 * (ts - 12)) / 24 # graus

        a = 0.4090 + (0.5016 * sin(radians(Ws - 60)))
        b = 0.6609 + (0.4767 * sin(radians(Ws - 60)))

        f_c = a + 0.5*b * ( ((pi*Ws/180) - sin(radians(Ws))*cos(radians(Ws))) / (sin(radians(Ws)) - (pi*Ws/180) * cos(radians(Ws))) )
        r_0 = (pi/24) * ( (cos(radians(W)) - cos(radians(Ws))) / (sin(radians(Ws)) - (pi*Ws/180) * cos(radians(Ws))) )

        return (((a + b*cos(radians(W))) * r_0) / f_c ) * (H * (1 + correction))
