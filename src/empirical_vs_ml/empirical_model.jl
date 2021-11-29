"""
Contém a implementação do modelo emírico 


l = [ (a + b*cos(W) * r_0) / f_c  ] * H

"""

module EmpiricalModels

export predict

"""
    Parametros
    -----------

    lat: Latitude em graus decimais
    lon: Latitude em graus decimais
    LT:  Local time (hora locaal no momento do registro)
    Ls:  Meridiano local (para Brasilia: -3)
    doy: Dia do ano (1o de Janeiro = 1, ... 31 de Dezembro)
    H:   Total de radiação solar no dia do registro
"""
function predict(lat::Float64, lon::Float64, LT::Int64, Ls::Int64, doy::Int64, H::Float64; correction::Float64 = 0.2) :: Float64

    B = (360 * (doy - 81)) / 365
    ET = (9.87 * sind(2*B)) - (7.53 * cosd(B)) - (1.5 * cosd(B))

    ts = LT + (ET/60) + (4/60)*(Ls - lon)

    G = rad2deg( (2 * pi * (doy - 1)) / 365 )
    gamma = (180 / pi) * ( 0.006918           - 0.399912*cosd(G)   +
                           0.070257*sind(G)   - 0.006758*cosd(2*G) + 
                           0.000907*sind(2*G) - 0.002697*cosd(3*G) +
                           0.001480*sind(3*G) )

    Ws = acosd( -tand(lat) * tand(gamma) )
    W = (360 * (ts - 12)) / 24

    a = 0.4090 + (0.5016 * sind(Ws - 60))
    b = 0.6609 + (0.4767 * sind(Ws - 60))

    f_c = a + 0.5*b * ( ((pi*Ws/180) - sind(Ws)*cosd(Ws)) / (sind(Ws) - (pi*Ws/180) * cosd(Ws)) )
    r_0 = (pi/24) * ( (cosd(W) - cosd(Ws)) / (sind(Ws) - (pi*Ws/180) * cosd(Ws)) )

    return ( ( (a + b*cosd(W)) * r_0) / f_c ) * (H * (1 + correction)) 
end

end
