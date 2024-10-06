import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from datetime import datetime
import math
from math import floor

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Constants
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Utility functions
def rev(angle):
    return angle - math.floor(angle / 360.0) * 360.0  # 0 <= a < 360

def rev2(angle):
    a = rev(angle)
    return a - 360.0 if a >= 180 else a  # -180 <= a < 180

def sind(angle):
    return math.sin(angle * DEG2RAD)

def cosd(angle):
    return math.cos(angle * DEG2RAD)

def tand(angle):
    return math.tan(angle * DEG2RAD)

def asind(value):
    return RAD2DEG * math.asin(value)

def acosd(value):
    return RAD2DEG * math.acos(value)

def atand(value):
    return RAD2DEG * math.atan(value)

def atan2d(y, x):
    return RAD2DEG * math.atan2(y, x)

def log10(x):
    return math.log10(x)

def sqr(x):
    return x * x

def cbrt(x):
    return x ** (1/3.0)

def SGN(x):
    return -1 if x < 0 else 1

def sunxyz(jday):
    d = jday - 2451543.5
    w = 282.9404 + 4.70935E-5 * d
    e = 0.016709 - 1.151E-9 * d
    M = rev(356.0470 + 0.9856002585 * d)
    E = M + e * RAD2DEG * sind(M) * (1.0 + e * cosd(M))
    xv = cosd(E) - e
    yv = math.sqrt(1.0 - e * e) * sind(E)
    v = atan2d(yv, xv)
    r = math.sqrt(xv * xv + yv * yv)
    lonsun = rev(v + w)
    xs = r * cosd(lonsun)
    ys = r * sind(lonsun)
    return [xs, ys, 0, r, lonsun, 0]

def radec2aa(ra, dec, jday, obs):
    TH0 = 280.46061837 + 360.98564736629 * (jday - 2451545.0)
    H = rev(TH0 - obs['longitude'] - ra)
    alt = asind(sind(obs['latitude']) * sind(dec) + cosd(obs['latitude']) * cosd(dec) * cosd(H))
    az = atan2d(sind(H), (cosd(H) * sind(obs['latitude']) - tand(dec) * cosd(obs['latitude'])))
    return [alt, rev(az + 180.0), H]

def SunAlt(jday, obs):
    sdat = sunxyz(jday)
    ecl = 23.4393 - 3.563E-7 * (jday - 2451543.5)
    xe = sdat[0]
    ye = sdat[1] * cosd(ecl)
    ze = sdat[1] * sind(ecl)
    ra = rev(atan2d(ye, xe))
    dec = atan2d(ze, math.sqrt(xe * xe + ye * ye))
    topo = radec2aa(ra, dec, jday, obs)
    return [topo[0], topo[1], topo[2], ra, dec, sdat[4], 0, 1, 0, sdat[3], -26.74]


def MoonPos(jday, obs):
    T = (jday - 2451545.0) / 36525
    LP = rev(218.3164477 + 481267.88123421 * T)
    D = rev(297.8501921 + 445267.1114034 * T)
    M = rev(357.5291092 + 35999.0502909 * T)
    MP = rev(134.9633964 + 477198.8675055 * T)
    F = rev(93.2720950 + 483202.0175233 * T)
    
    Sl = (6288774 * sind(MP) + 1274027 * sind(2 * D - MP) + 658314 * sind(2 * D) +
          213618 * sind(2 * MP) - 185116 * sind(M) - 114332 * sind(2 * F) +
          58793 * sind(2 * D - 2 * MP) + 57066 * sind(2 * D - M - MP) +
          53322 * sind(2 * D + MP) + 45758 * sind(2 * D - M) -
          40923 * sind(M - MP) - 34720 * sind(D) - 30383 * sind(M + MP) +
          15327 * sind(2 * D - 2 * F) - 12528 * sind(MP + 2 * F) +
          10980 * sind(MP - 2 * F) + 10675 * sind(4 * D - MP) +
          10034 * sind(3 * MP) + 8548 * sind(4 * D - 2 * MP) -
          7888 * sind(2 * D + M - MP) - 6766 * sind(2 * D + M) -
          5163 * sind(D - MP) + 4987 * sind(D + M) + 4036 * sind(2 * D - M + MP))
    
    Sb = (5128122 * sind(F) + 280602 * sind(MP + F) + 277602 * sind(MP - F) +
          173237 * sind(2 * D - F) + 55413 * sind(2 * D - MP + F) +
          46271 * sind(2 * D - MP - F) + 32573 * sind(2 * D + F) +
          17198 * sind(2 * MP + F) + 9266 * sind(2 * D + MP - F) +
          8822 * sind(2 * MP - F) + 8216 * sind(2 * D - M - F) +
          4324 * sind(2 * D - 2 * MP - F) + 4200 * sind(2 * D + MP + F))
    
    Sr = ((-20905355) * cosd(MP) - 3699111 * cosd(2 * D - MP) -
          2955968 * cosd(2 * D) - 569925 * cosd(2 * MP) +
          246158 * cosd(2 * D - 2 * MP) - 152138 * cosd(2 * D - M - MP) -
          170733 * cosd(2 * D + MP) - 204586 * cosd(2 * D - M) -
          129620 * cosd(M - MP) + 108743 * cosd(D) + 104755 * cosd(M + MP) +
          79661 * cosd(MP - 2 * F) + 48888 * cosd(M))
    
    mglong = rev(LP + Sl / 1000000.0)
    mglat = Sb / 1000000.0
    obl = 23.4393 - 3.563E-7 * (jday - 2451543.5)
    ra = rev(atan2d(sind(mglong) * cosd(obl) - tand(mglat) * sind(obl), cosd(mglong)))
    dec = asind(sind(mglat) * cosd(obl) + cosd(mglat) * sind(obl) * sind(mglong))
    moondat = radec2aa(ra, dec, jday, obs)
    pa = abs(180.0 - D - 6.289 * sind(MP) + 2.100 * sind(M) - 1.274 * sind(2 * D - MP) -
             0.658 * sind(2 * D) - 0.214 * sind(2 * MP) - 0.11 * sind(D))
    k = (1 + cosd(pa)) / 2
    mr = round(385000.56 + Sr / 1000.0)
    h = moondat[0]
    h -= asind(6378.14 / mr) * cosd(h)
    sdat = sunxyz(jday)
    r = sdat[3]
    R = mr / 149598000
    mag = 0.23 + 5 * log10(r * R) + 0.026 * pa + 4.0E-9 * pa * pa * pa * pa
    return [h, moondat[1], moondat[2], ra, dec, mglong, mglat, k, mag, mr, -12.7]

def to_julian_date(dt):
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    C = math.floor(365.25 * year)
    D = math.floor(30.6001 * (month + 1))

    jd = B + C + D + day + 1720994.5
    return jd

# Constants
GREGORIAN_EPOCH = 1721425.5
d2r = math.pi / 180
r2d = 180 / math.pi

# Constants
MERCURY = 3
VENUS = 5
EARTH = 2
MARS = 2
JUPITER = 4
SATURN = 6
SUN = 0
MOON = 1

class Planet:
    def __init__(self, name, num, N, i, w, a, e, M):
        self.name = name
        self.num = num
        self.N = N  # longitude of ascending node
        self.i = i  # inclination
        self.w = w  # argument of perihelion
        self.a = a  # semimajor axis
        self.e = e  # eccentricity
        self.M = M  # mean anomaly

# Initialize planets with data
planets = [None] * 7
planets[MERCURY] = Planet("Mercury", 0, [48.3313, 3.24587E-5], [7.0047, 5.00E-8], [29.1241, 1.01444E-5], [0.387098, 0], [0.205635, 5.59E-10], [168.6562, 4.0923344368])
planets[VENUS] = Planet("Venus", 1, [76.6799, 2.46590E-5], [3.3946, 2.75E-8], [54.8910, 1.38374E-5], [0.723330, 0], [0.006773, -1.302E-9], [48.0052, 1.6021302244])
planets[MARS] = Planet("Mars", 3, [49.5574, 2.11081E-5], [1.8497, -1.78E-8], [286.5016, 2.92961E-5], [1.523688, 0], [0.093405, 2.516E-9], [18.6021, 0.5240207766])
planets[JUPITER] = Planet("Jupiter", 4, [100.4542, 2.76854E-5], [1.3030, -1.557E-7], [273.8777, 1.64505E-5], [5.20256, 0], [0.048498, 4.469E-9], [19.8950, 0.0830853001])
planets[SATURN] = Planet("Saturn", 5, [113.6634, 2.38980E-5], [2.4886, -1.081E-7], [339.3939, 2.97661E-5], [9.55475, 0], [0.055546, -9.499E-9], [316.9670, 0.0334442282])


# Functions for planetary positions and calculations
def helios(p, jday):
    d = jday - 2451543.5
    w = p.w[0] + p.w[1] * d  # argument of perihelion
    e = p.e[0] + p.e[1] * d
    a = p.a[0] + p.a[1] * d
    i = p.i[0] + p.i[1] * d
    N = p.N[0] + p.N[1] * d
    M = rev(p.M[0] + p.M[1] * d)  # mean anomaly
    E0 = M + RAD2DEG * e * sind(M) * (1.0 + e * cosd(M))
    E1 = E0 - (E0 - RAD2DEG * e * sind(E0) - M) / (1.0 - e * cosd(E0))
    while abs(E0 - E1) > 0.0005:
        E0 = E1
        E1 = E0 - (E0 - RAD2DEG * e * sind(E0) - M) / (1.0 - e * cosd(E0))
    xv = a * (cosd(E1) - e)
    yv = a * math.sqrt(1.0 - e * e) * sind(E1)
    v = rev(atan2d(yv, xv))  # true anomaly
    r = math.sqrt(xv * xv + yv * yv)  # distance
    xh = r * (cosd(N) * cosd(v + w) - sind(N) * sind(v + w) * cosd(i))
    yh = r * (sind(N) * cosd(v + w) + cosd(N) * sind(v + w) * cosd(i))
    zh = r * (sind(v + w) * sind(i))
    lonecl = atan2d(yh, xh)
    latecl = atan2d(zh, math.sqrt(xh * xh + yh * yh + zh * zh))

    if p.num == JUPITER:  # Jupiter perturbations by Saturn
        Ms = rev(planets[SATURN].M[0] + planets[SATURN].M[1] * d)
        lonecl += (-0.332) * sind(2 * M - 5 * Ms - 67.6) - 0.056 * sind(2 * M - 2 * Ms + 21) + 0.042 * sind(3 * M - 5 * Ms + 21) - 0.036 * sind(M - 2 * Ms) + 0.022 * cosd(M - Ms) + 0.023 * sind(2 * M - 3 * Ms + 52) - 0.016 * sind(M - 5 * Ms - 69)
        xh = r * cosd(lonecl) * cosd(latecl)  # recalc xh, yh
        yh = r * sind(lonecl) * cosd(latecl)

    if p.num == SATURN:  # Saturn perturbations
        Mj = rev(planets[JUPITER].M[0] + planets[JUPITER].M[1] * d)
        lonecl += 0.812 * sind(2 * Mj - 5 * M - 67.6) - 0.229 * cosd(2 * Mj - 4 * M - 2) + 0.119 * sind(Mj - 2 * M - 3) + 0.046 * sind(2 * Mj - 6 * M - 69) + 0.014 * sind(Mj - 3 * M + 32)
        latecl += -0.020 * cosd(2 * Mj - 4 * M - 2) + 0.018 * sind(2 * Mj - 6 * M - 49)
        xh = r * cosd(lonecl) * cosd(latecl)  # recalc xh, yh, zh
        yh = r * sind(lonecl) * cosd(latecl)
        zh = r * sind(latecl)

    return [xh, yh, zh, r, lonecl, latecl]
def radecr(obj, sun, jday, obs):
    xg = obj[0] + sun[0]
    yg = obj[1] + sun[1]
    zg = obj[2]
    obl = 23.4393 - 3.563E-7 * (jday - 2451543.5)
    x1 = xg
    y1 = yg * cosd(obl) - zg * sind(obl)
    z1 = yg * sind(obl) + zg * cosd(obl)
    ra = rev(atan2d(y1, x1))
    dec = atan2d(z1, math.sqrt(x1 * x1 + y1 * y1))
    dist = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    return [ra, dec, dist]

def PlanetAlt(p, jday, obs):
    # Alt/Az, hour angle, ra/dec, ecliptic long. and lat, illuminated fraction, dist(Sun), dist(Earth), brightness of planet p
    if p == 0:
        return SunAlt(jday, obs)
    if p == 1:
        return MoonPos(jday, obs)
    
    sun_xyz = sunxyz(jday)
    planet_xyz = helios(planets[p], jday)

    dx = planet_xyz[0] + sun_xyz[0]
    dy = planet_xyz[1] + sun_xyz[1]
    dz = planet_xyz[2] + sun_xyz[2]
    lon = rev(atan2d(dy, dx))
    lat = atan2d(dz, math.sqrt(dx * dx + dy * dy))

    radec = radecr(planet_xyz, sun_xyz, jday, obs)  # Pass 'obs' to 'radecr' if needed
    ra = radec[0]
    dec = radec[1]
    altaz = radec2aa(ra, dec, jday, obs)

    dist = radec[2]
    R = sun_xyz[3]  # Sun-Earth distance
    r = planet_xyz[3]  # heliocentric distance
    k = ((r + dist) * (r + dist) - R * R) / (4 * r * dist)  # illuminated fraction (41.2)

    # brightness calc according to Meeus p. 285-86 using Astronomical Almanac expressions
    absbr = [-0.42, -4.40, 0, -1.52, -9.40, -8.88, -7.19, -6.87]
    i = math.degrees(math.acos((r * r + dist * dist - R * R) / (2 * r * dist)))  # phase angle
    mag = absbr[p] + 5 * math.log10(r * dist)  # common for all planets
    if p == 0:
        mag += i * (0.0380 + i * (-0.000273 + i * 0.000002))
    elif p == 1:
        mag += i * (0.0009 + i * (0.000239 - i * 0.00000065))
    elif p == 3:
        mag += i * 0.016
    elif p == 4:
        mag += i * 0.005
    elif p == 5:
        T = (jday - 2451545.0) / 36525
        incl = 28.075216 - 0.012998 * T + 0.000004 * T * T
        omega = 169.508470 + 1.394681 * T + 0.000412 * T * T
        B = math.degrees(math.asin(math.sin(math.radians(incl)) * math.cos(math.radians(lat)) * math.sin(math.radians(lon - omega)) - math.cos(math.radians(incl)) * math.sin(math.radians(lat))))
        l = planet_xyz[4]  # heliocentric longitude of Saturn
        b = planet_xyz[5]  # heliocentric latitude
        U1 = math.degrees(math.atan2(math.sin(math.radians(incl)) * math.sin(math.radians(b)) + math.cos(math.radians(incl)) * math.cos(math.radians(b)) * math.sin(math.radians(l - omega)), math.cos(math.radians(b)) * math.cos(math.radians(l - omega))))
        U2 = math.degrees(math.atan2(math.sin(math.radians(incl)) * math.sin(math.radians(lat)) + math.cos(math.radians(incl)) * math.cos(math.radians(lat)) * math.sin(math.radians(lon - omega)), math.cos(math.radians(lat)) * math.cos(math.radians(lon - omega))))
        dU = abs(U1 - U2)
        mag += 0.044 * dU - 2.60 * math.sin(math.radians(abs(B))) + 1.25 * math.sin(math.radians(B)) * math.sin(math.radians(B))

    return [altaz[0], altaz[1], altaz[2], ra, dec, lon, lat, k, r, dist, mag]

def getGrahas(j, l, lat):
    obs = {'longitude': l, 'latitude': lat}
    grahas = [None] * 10
    grahas_next = [None] * 9
    speed = [None] * 9
    gr = [None] * 9
    grn = [None] * 9
    day = 1000 * 60 * 60 * 24

    for a in range(7):
        gr[a] = PlanetAlt(a, j, obs)
        if gr[a] is not None:
            grahas[a] = gr[a][5]
        grn[a] = PlanetAlt(a, j + 1, obs)
        if grn[a] is not None:
            grahas_next[a] = grn[a][5]
        if grahas[a] is not None and grahas_next[a] is not None:
            speed[a] = ((grahas_next[a] - grahas[a] + 360) % 360 / day
                        if (grahas_next[a] - grahas[a]) < -300
                        else (grahas_next[a] - grahas[a]) % 360 / day)

    return grahas, grahas_next, speed

def leap_gregorian(year):
    return (year % 4) == 0 and not ((year % 100) == 0 and (year % 400) != 0)

def fix360(v):
    if v < 0:
        v += 360
    if v > 360:
        v -= 360
    return v

def calcAyanamsa(t):
    ln = ((933060 - 6962911 * t + 7.5 * t * t) / 3600.0) % 360.0
    off = (259205536.0 * t + 2013816.0) / 3600.0
    off = 17.23 * sind(ln) + 1.27 * sind(off) - (5025.64 + 1.11 * t) * t
    off = (off - 80861.27) / 3600.0
    ayanamsa = off
    node = (ln + off + 360) % 360
    return ayanamsa, node

def calculateAscendant(date_time, j, latitude, longitude, timezone_offset):
    hr = date_time.hour + date_time.minute / 60
    tz = timezone_offset
    f = hr + tz
    t = (j - 2415020) / 36525
    ayanamsa, node = calcAyanamsa(t)
    ra = (((6.6460656 + 2400.0512617 * t + 2.581e-5 * t * t + f) * 15 - longitude) % 360) * d2r
    ob = (23.452294 - 0.0130125 * t - 0.00000164 * t * t + 0.000000503 * t * t * t) * d2r

    mc = math.atan2(math.tan(ra), math.cos(ob))
    if mc < 0:
        mc += math.pi
    if math.sin(ra) < 0:
        mc += math.pi
    mc *= r2d

    as_ = math.atan2(math.cos(ra), -math.sin(ra) * math.cos(ob) - math.tan(latitude * d2r) * math.sin(ob))
    if as_ < 0:
        as_ += math.pi
    if math.cos(ra) < 0:
        as_ += math.pi
    as_ = fix360(as_ * r2d)

    as_ = fix360(as_ + ayanamsa)
    mc = fix360(mc + ayanamsa)

    hs = [None] * 24
    x = as_ - mc
    if x < 0:
        x += 360
    x /= 6
    y = 18
    for i in range(7):
        hs[y % 24] = mc + x * i
        y += 1
        if y > 24:
            y = 0

    x = mc - fix360(as_ + 180)
    if x < 0:
        x += 360
    x /= 6
    y = 12
    for i in range(7):
        hs[y] = fix360(as_ + 180 + x * i)
        y += 1

    for i in range(12):
        hs[i] = fix360(hs[i + 12] + 180)

    bhaava_madya = [
        hs[0], hs[2], hs[4], hs[6], hs[8], hs[10],
        hs[12], hs[14], hs[16], hs[18], hs[20], hs[22]
    ]
    bhaava_sandhi = [
        hs[1], hs[3], hs[5], hs[7], hs[9], hs[11],
        hs[13], hs[15], hs[17], hs[19], hs[21], hs[23]
    ]

    return as_, mc, bhaava_madya, bhaava_sandhi, ayanamsa, node

def m2j(dob):
    m = dob.month
    d = dob.day
    y = dob.year
    sec = dob.second
    min = dob.minute
    hour = dob.hour

    # Calculate Julian Day Number
    julian_day = (GREGORIAN_EPOCH - 1) + \
                 (365 * (y - 1)) + \
                 math.floor((y - 1) / 4) + \
                 -math.floor((y - 1) / 100) + \
                 math.floor((y - 1) / 400) + \
                 math.floor((((367 * m) - 362) / 12) + \
                 (0 if m <= 2 else -1 if leap_gregorian(y) else -2) + \
                 d) + \
                 (math.floor(sec + 60 * (min + 60 * hour) + 0.5) / 86400.0)
    
    return julian_day

def j2j(JulianDay):
    j = int(JulianDay) + 1402
    k = (j - 1) // 1461
    l = j - 1461 * k
    n = (l - 1) // 365 - l // 1461
    i = l - 365 * n + 30
    J = int(80 * i / 2447)
    I2 = J // 11
    day0 = i - int(2447 * J / 80)
    month = J + 2 - 12 * I2
    year = 4 * k + n + I2 - 4716
    return day0, month, year

def j2g(JulianDay):
    a = JulianDay + 68569
    b = int(a / 36524.25)
    c = a - int(36524.25 * b + 0.75)
    e = int((c + 1) / 365.2425)
    f = c - int(365.25 * e) + 31
    g = int(f / 30.59)
    h = g // 11
    day0 = math.floor(f - int(30.59 * g) + (JulianDay - int(JulianDay)))
    month = math.floor(g - 12 * h + 2)
    year = math.floor(100 * (b - 49) + e + h)
    return day0, month, year

def jd2md2(j):
    if j < 2299239:
        day0, month, year = j2j(j)
    else:
        day0, month, year = j2g(j)
    hour = (j - int(j)) * 24
    minute = (hour - int(hour)) * 60
    second = (minute - int(minute)) * 60
    return datetime(year, month, day0, int(hour), int(minute), int(second))
RASHI_NAMES = [
    'Mesha (Aries)', 'Vrishabha (Taurus)', 'Mithuna (Gemini)', 'Karka (Cancer)',
    'Simha (Leo)', 'Kanya (Virgo)', 'Tula (Libra)', 'Vrishchika (Scorpio)',
    'Dhanu (Sagittarius)', 'Makara (Capricorn)', 'Kumbha (Aquarius)', 'Meena (Pisces)'
]

def fra(t):
    return t - floor(t)

def tt(t, y):
    if y == 0:
        z = f"{t:02d}" if t < 10 else str(t)
    else:
        if t > 100:
            z = str(t)
        elif t > 10:
            z = f"{t:02d}"
        else:
            z = f"{t:03d}"
    return z
w=["Ar","Ta","Ge","Ca","Le","Vi","Li","Sc","Sg","Cp","aq","Pi"]
def todeg(t):
    r = fra(t)
    a = floor(t)
    r2 = r * 60
    return f"{tt(a % 30)} {w[floor(a / 30)]} {tt(floor(r2))}' {tt(floor(fra(r2) * 60))}\""

def wrapT(ctx, t, x, y, mW, lH):
    w = t.split('')
    l = ''
    for n in range(len(w)):
        tL = l + w[n]
        m = ctx.measureText(tL)
        tW = m.width
        if tW > mW and n > 0:
            ctx.fillText(l, x, y)
            l = w[n]
            y += lH
        else:
            l = tL
    ctx.fillText(l, x, y)

def reddeg(d):
    return (d + 360) % 360

def getRasi(d):
    return floor(reddeg(d) / 30)

def red_a(g, a):
    return g % a

def reddeg(g):
    return g % 360

def get_rasi_len(g):
    return red_a(g, 30)

def is_odd_rasi(g):
    return get_rasi(g + 30) % 2

def is_even_rasi(g):
    return get_rasi(g) % 2

def get_dvadasamsa_length(g):
    return reddeg(get_rasi(g) * 30 + get_rasi_len(g) * 12)

def in_movable_sign(g):
    return get_rasi(g) % 3 == 0

def in_fixed_sign(g):
    return get_rasi(g) % 3 == 1

def in_dual_sign(g):
    return get_rasi(g) % 3 == 2

def get_rasi(g):
    return int(reddeg(g) / 30)

def naks(g):
    return floor(g / (40 / 3)) % 27

def vargas(d, grahas):
    v = []
    p = grahas[:]
    t = []

    for i in range(10):
        if d == 1:
            p[i] = p[i] * d
        elif d == 2:
            p[i] = red_a(p[i] - 15, 60) + 90
        elif d == 3:
            p[i] = (int(get_rasi_len(p[i]) / 10) * 120 + get_rasi(p[i]) * 30 + get_rasi_len(d * p[i]))
        elif d == 4:
            p[i] = (int(get_rasi_len(p[i]) / 7.5) * 90 + get_rasi(p[i]) * 30 + get_rasi_len(d * p[i]))
        elif d == 6:
            p[i] = d * p[i]
        elif d == 7:
            t.append(get_rasi(p[i]) * 30 + get_rasi_len(p[i]) * d)
            if is_odd_rasi(p[i]):
                p[i] = t[i]
            else:
                p[i] = t[i] + 180
        elif d == 8:
            p[i] = d * p[i]
        elif d == 9:
            p[i] = d * p[i]
        elif d == 10:
            t.append(get_rasi(p[i]) * 30 + get_rasi_len(p[i]) * d)
            if is_odd_rasi(p[i]):
                p[i] = t[i]
            else:
                p[i] = t[i] + 240
        elif d == 12:
            p[i] = get_dvadasamsa_length(p[i])
        elif d == 16:
            p[i] = d * p[i]
        elif d == 20:
            p[i] = d * p[i]
        elif d == 24:
            t.append(get_rasi_len(p[i]) * d)
            if is_odd_rasi(p[i]):
                p[i] = t[i] + 120
            else:
                p[i] = t[i] + 90
        elif d == 27:
            p[i] = d * p[i]
        elif d == 30:
            t.append(get_rasi_len(p[i]))
            if is_odd_rasi(p[i]):
                if t[i] < 5:
                    p[i] = d * 0 + t[i] * 6
                elif 5 <= t[i] <= 10:
                    p[i] = d * 10 + (t[i] - 5) * 6
                elif 10 <= t[i] <= 18:
                    p[i] = d * 8 + (t[i] - 10) / 4 * 15
                elif 18 <= t[i] <= 25:
                    p[i] = d * 2 + (t[i] - 18) / 7 * d
                else:
                    p[i] = d * 6 + (t[i] - 25) * 6
            else:
                if t[i] < 5:
                    p[i] = d * 1 + (5 - t[i]) * 6
                elif 5 <= t[i] <= 10:
                    p[i] = d * 5 + (10 - t[i]) * 6
                elif 10 <= t[i] <= 18:
                    p[i] = d * 11 + (18 - t[i]) / 4 * 15
                elif 18 <= t[i] <= 25:
                    p[i] = d * 9 + (25 - t[i]) / 7 * d
                else:
                    p[i] = d * 7 + (d - t[i]) * 6
        elif d == 40:
            t.append(get_rasi_len(p[i]) * d)
            if is_odd_rasi(p[i]):
                p[i] = t[i]
            else:
                p[i] = t[i] + 180
        elif d == 45:
            t.append(get_rasi_len(p[i]) * d)
            if in_movable_sign(p[i]):
                p[i] = t[i]
            elif in_fixed_sign(p[i]):
                p[i] = t[i] + 120
            else:
                p[i] = t[i] + 240
        elif d == 60:
            p[i] = d * get_rasi_len(p[i]) + get_rasi(p[i]) * 30
        elif d == 108:
            p[i] = get_dvadasamsa_length(9 * p[i])
        elif d == 144:
            p[i] = get_dvadasamsa_length(get_dvadasamsa_length(p[i]))
        else:
            p[i] = d * p[i]
        v.append(get_rasi(p[i]))

    return v

def varga_bhaava(d):
    vb = [(12 + d[i] - d[8]) % 12 for i in range(10)]
    return vb
    
def vChart(d, e):
  
    d = [0,1,10,11,7,1,3,10,9,4]
    if isinstance(e, (int, float)):
        # Debugging: print d to check the values
        n = ['सू', 'चं', 'मं', 'बु', 'बृ', 'शु', 'श', 'रा', 'ल', 'के']
        f = [0,0,0, 0, 1,0,0,1,0,1 ]
        g = 0
        s = [""] * 13  # Create an array with 13 empty strings

        # for i in range(10):
        #     k = int(d[i] + 1)
            
        #     # Fix the string assignment logic
        #     s[k] = (s[k] if s[k] != "" else "") + n[i] + " "
        
        # # Cleaning up the results (if necessary)
        # for i in range(13):
        #     # This line is redundant but kept for consistency
        #     s[i] = s[i].replace('undefined ', '').replace('undefined', '')
        for i in range(10):
            k = d[i] + 1
            
            if s[k] == "":
                s[k] = " "
            else:
                s[k] = s[k] + " "
            
            if f[i] == 1 and g == 1:
                s[k] += f"({{{n[i]}}})"
            elif f[i] == 1:
                s[k] += f"({n[i]})"
            elif g == 1:
                s[k] += f"{{{n[i]}}}"
            else:
                s[k] += str(n[i])

        for i in range(13):
            s[i] = str(s[i])  # Ensures s[i] is a string
            s[i] = s[i].replace('undefined ', '').replace('undefined', '')
                
        print(d,f,g,e,"joshi")
        print(s)
        return s  # Return the constructed array of strings

as_rashi = [
    { 'f': "मेष        /Aries", 'onme': 'Kriya', 't': 'मे/    Ar', 's': 'mars', 'n': 2 },
    { 'f': "बृषभ     /Taurus", 'onme': 'Thavuri', 't': 'बृ/    Ta', 's': 'venus', 'n': 5 },
    { 'f': "मिथुन    /Gemini", 'onme': 'Jitheema', 't': 'मि/   Ge', 's': 'mercury', 'n': 3 },
    { 'f': "कर्क     /Cancer", 'onme': 'Kulira', 't': 'क/   Ca', 's': 'moon', 'n': 1 },
    { 'f': "सिंह        /Leo", 'onme': 'Laya', 't': 'सिं/  Le', 's': 'sun', 'n': 0 },
    { 'f': "कन्या     /Virgo", 'onme': 'Pathona', 't': 'कन्/ Vi', 's': 'mercury', 'n': 3 },
    { 'f': "तुला      /Libra", 'onme': 'Juka', 't': 'तु /   Li', 's': 'venus', 'n': 5 },
    { 'f': "वृश्चिक/Scorpius", 'onme': 'Kowrpi', 't': 'वृ/    Sc', 's': 'mars', 'n': 2 },
    { 'f': "धनु /Sagittarius", 'onme': 'Thaukshika', 't': 'धु/    Sg', 's': 'jupiter', 'n': 4 },
    { 'f': "मकर/Capricorns", 'onme': 'Akokero', 't': 'म/    Cp', 's': 'saturn', 'n': 6 },
    { 'f': "कुंभ  /Aquarius", 'onme': 'Hridroga', 't': 'कुं/   Aq', 's': 'saturn', 'n': 6 },
    { 'f': "मीन   /Pisces", 'onme': 'Anthya', 't': 'मी/   Pi', 's': 'jupiter', 'n': 4 }
]

def ansh5(g):
    return int(reddeg(g) / 5)

def Bhaava(g, grahas):
    return (int(g / 30) - int(grahas[8] / 30) + 12) % 12

def naks(g):
    return int(g / (40 / 3)) % 27

def lordOf(b, grahas):  # get lord of a bhaava (input is bhaava)
    # return graha swami no. for bhaava input
    s = get_rasi(grahas[8])
    l = red12(s + b)
    return as_rashi[l]['n']

def isPlanetInHouse(g, a, grahas):  # check whether a planet is in a particular house
    # s = lord_of(gra)
    s2 = Bhaava(grahas[g], grahas)
    return a == s2


def getLord(r):  # get lord of a rasi (input is rasi no. - 1)
    return as_rashi[r].n


def getGrahaDrishti(g, a):
    if a == 6:
        return True  # || a == -6
    elif g == 2 and (a in [-5, -9, 8, 3]):  # mars - g=2 drishti 4 and 8
        return True
    elif g == 4 and (a in [-4, -8, 8, 4]):  # jupiter - g=4 drishti 5 and 9
        return True
    elif g == 6 and (a in [-3, -10, 9, 2]):  # saturn - g=6 drishti 3 and 10
        return True
    else:
        return False


def isGrahaDrishti(g, d, b):
    a = red12(b - d)
    # if d < 0: d = 12 + red12(a - b)
    return getGrahaDrishti(g, a)


# Function to normalize degrees between 0 and 360
def normalize_degrees(degrees):
    return (degrees % 360 + 360) % 360

# Function to calculate Rasi (sign) based on degrees
def get_rasi_from_degrees(degrees):
    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 
             'Aquarius', 'Pisces']
    return signs[int(degrees // 30)]

# Function to calculate Hora (D-2 Rasi)
def get_rasi_d2(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    sign_degree = degrees % 30
    if sign_index % 2 == 0:  # Even sign
        return 'Moon' if sign_degree < 15 else 'Sun'
    else:  # Odd sign
        return 'Sun' if sign_degree < 15 else 'Moon'

# Function to calculate D-9 Rasi
def getRasiD9(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    navamsa_index = int((degrees % 30) // (30 / 9))
    rasi_sequence = [
      [0, 1, 2, 3, 4, 5, 6, 7, 8],  
        [9, 10, 11, 0, 1, 2, 3, 4, 5], 
        [6, 7, 8, 9, 10, 11, 0, 1, 2],  
        [3, 4, 5, 6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  
        [9, 10, 11, 0, 1, 2, 3, 4, 5], 
        [6, 7, 8, 9, 10, 11, 0, 1, 2],  
        [3, 4, 5, 6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  
        [9, 10, 11, 0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11, 0, 1, 2],  
        [3, 4, 5, 6, 7, 8, 9, 10, 11]   
    ]
    return rasi_sequence[sign_index][navamsa_index]

# Function to calculate D-7 Rasi
def getRasiD7(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    saptamsa_index = int((degrees % 30) // (30 / 7))
    
    rasi_sequence_d7 = [
         [0, 1, 2, 3, 4, 5, 6],  
        [7, 8, 9, 10, 11, 0, 1],  
        [2, 3, 4, 5, 6, 7, 8],  
        [9, 10, 11, 0, 1, 2, 3],  
        [4, 5, 6, 7, 8, 9, 10],  
        [11, 0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11, 0],  
        [1, 2, 3, 4, 5, 6, 7], 
        [8, 9, 10, 11, 0, 1, 2],  
        [3, 4, 5, 6, 7, 8, 9], 
        [10, 11, 0, 1, 2, 3, 4],  
        [5, 6, 7, 8, 9, 10, 11]   
    ]
    return rasi_sequence_d7[sign_index][saptamsa_index]

# Function to calculate D-10 Rasi
def getRasiD10(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    dashamsa_index = int((degrees % 30) // (30 / 10))
    
    rasi_sequence_d10 = [
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  
        [9, 10, 11, 0, 1, 2, 3, 4, 5, 6],  
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        [11, 0, 1, 2, 3, 4, 5, 6, 7, 8],  
        [4, 5, 6, 7, 8, 9, 10, 11, 0, 1], 
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        [6, 7, 8, 9, 10, 11, 0, 1, 2, 3],  
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 0],  
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5], 
        [5, 6, 7, 8, 9, 10, 11, 0, 1, 2],  
        [10, 11, 0, 1, 2, 3, 4, 5, 6, 7],  
        [7, 8, 9, 10, 11, 0, 1, 2, 3, 4]   
    ]
    return rasi_sequence_d10[sign_index][dashamsa_index]

# Function to calculate D-8 Rasi
def getRasiD8(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    ashtamsa_index = int((degrees % 30) // (30 / 8))
    
    rasi_sequence_d8 = [
        [0, 1, 2, 3, 4, 5, 6, 7],  
        [8, 9, 10, 11, 0, 1, 2, 3], 
        [4, 5, 6, 7, 8, 9, 10, 11], 
        [0, 1, 2, 3, 4, 5, 6, 7], 
        [8, 9, 10, 11, 0, 1, 2, 3],  
        [4, 5, 6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5, 6, 7],  
        [8, 9, 10, 11, 0, 1, 2, 3],  
        [4, 5, 6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5, 6, 7],  
        [8, 9, 10, 11, 0, 1, 2, 3],  
        [4, 5, 6, 7, 8, 9, 10, 11]  
    ]
    return rasi_sequence_d8[sign_index][ashtamsa_index]

# Function to calculate D-6 Rasi
def getRasiD6(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    shashtamsa_index = int((degrees % 30) // (30 / 6))
    
    rasi_sequence_d6 = [
       [0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5], 
        [6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5], 
        [6, 7, 8, 9, 10, 11], 
        [0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11]  
    ]
    return rasi_sequence_d6[sign_index][shashtamsa_index]

def normalize_degrees(degrees):
    return degrees % 360  # Normalize degrees between 0 and 360

def getRasiD4(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    chaturthamsa_index = int((degrees % 30) // (30 / 4))
    
    rasi_sequence_d4 = [
       [0, 3, 6, 9],  
        [1, 4, 7, 10],  
        [2, 5, 8, 11],  
        [3, 6, 9, 0],  
        [4, 7, 10, 1],  
        [5, 8, 11, 2],  
        [6, 9, 0, 3],  
        [7, 10, 1, 4],  
        [8, 11, 2, 5],  
        [9, 0, 3, 6],  
        [10, 1, 4, 7],  
        [11, 2, 5, 8]  
    ]
    
    return rasi_sequence_d4[sign_index][chaturthamsa_index]

def getRasiD30(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    d30_degrees = degrees % 30
    
    odd_sign_sequence = [
        {'lord': 0, 'length': 5},  # Mars - 5°
        {'lord': 10, 'length': 5},  # Saturn - 5°
        {'lord': 8, 'length': 8},  # Jupiter - 8°
        {'lord': 2, 'length': 7},  # Mercury - 7°
        {'lord': 7, 'length': 5}   # Venus - 5°
    ]
    
    even_sign_sequence = [
        {'lord': 1, 'length': 5},  # Venus - 5°
        {'lord': 5, 'length': 7},  # Mercury - 7°
        {'lord': 11, 'length': 8},  # Jupiter - 8°
        {'lord': 9, 'length': 5},  # Saturn - 5°
        {'lord': 6, 'length': 5}   # Mars - 5°
    ]
    
    is_odd_sign = sign_index % 2 == 0
    trimsamsa_sequence = odd_sign_sequence if is_odd_sign else even_sign_sequence

    accumulated_degrees = 0
    for division in trimsamsa_sequence:
        accumulated_degrees += division['length']
        if d30_degrees < accumulated_degrees:
            return division['lord']

def getRasiD27(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    d27_index = int((degrees % 30) // (30 / 27))
    
    rasi_sequence_d27 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2],  
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
        [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2],  
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8],  
        [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2],  
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],  
        [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8],  
        [9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  
    ]
    
    return rasi_sequence_d27[sign_index][d27_index]

def getRasiD16(degrees):
    degrees = normalize_degrees(degrees)
    sign_index = int(degrees // 30)
    d16_index = int((degrees % 30) // (30 / 16))
    
    rasi_sequence_d16 = [
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3],  
        [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],  
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3], 
        [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],  
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3],  
        [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],  
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3], 
        [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],  
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
       
    ]
    
    return rasi_sequence_d16[sign_index][d16_index]



def pha(grahas , ascendant ):
    g = grahas.copy()
    suryaD9 = getRasiD9(g[0])
    chandraD9 = getRasiD9(g[1])
    mangalD9 = getRasiD9(g[2])
    budhD9 = getRasiD9(g[3])
    guruD9 = getRasiD9(g[4])
    shukraD9 = getRasiD9(g[5])
    shaniD9 = getRasiD9(g[6])
    rahuD9 = getRasiD9(g[7])
    ketuD9 = getRasiD9(g[7] + 180)
    lagnaD9 = getRasiD9(ascendant)

    suryaD4 = getRasiD4(g[0])
    chandraD4 = getRasiD4(g[1])
    mangalD4 = getRasiD4(g[2])
    budhD4 = getRasiD4(g[3])
    guruD4 = getRasiD4(g[4])
    shukraD4 = getRasiD4(g[5])
    shaniD4 = getRasiD4(g[6])
    rahuD4 = getRasiD4(g[7])
    ketuD4 = getRasiD4(g[7] + 180)
    lagnaD4 = getRasiD4(ascendant)

    suryaD6 = getRasiD6(g[0])
    chandraD6 = getRasiD6(g[1])
    mangalD6 = getRasiD6(g[2])
    budhD6 = getRasiD6(g[3])
    guruD6 = getRasiD6(g[4])
    shukraD6 = getRasiD6(g[5])
    shaniD6 = getRasiD6(g[6])
    rahuD6 = getRasiD6(g[7])
    ketuD6 = getRasiD6(g[7] + 180)
    lagnaD6 = getRasiD6(ascendant)

    suryaD7 = getRasiD7(g[0])
    chandraD7 = getRasiD7(g[1])
    mangalD7 = getRasiD7(g[2])
    budhD7 = getRasiD7(g[3])
    guruD7 = getRasiD7(g[4])
    shukraD7 = getRasiD7(g[5])
    shaniD7 = getRasiD7(g[6])
    rahuD7 = getRasiD7(g[7])
    ketuD7 = getRasiD7(g[7] + 180)
    lagnaD7 = getRasiD7(ascendant)

    suryaD8 = getRasiD8(g[0])
    chandraD8 = getRasiD8(g[1])
    mangalD8 = getRasiD8(g[2])
    budhD8 = getRasiD8(g[3])
    guruD8 = getRasiD8(g[4])
    shukraD8 = getRasiD8(g[5])
    shaniD8 = getRasiD8(g[6])
    rahuD8 = getRasiD8(g[7])
    ketuD8 = getRasiD8(g[7] + 180)
    lagnaD8 = getRasiD8(ascendant)

    suryaD10 = getRasiD10(g[0])
    chandraD10 = getRasiD10(g[1])
    mangalD10 = getRasiD10(g[2])
    budhD10 = getRasiD10(g[3])
    guruD10 = getRasiD10(g[4])
    shukraD10 = getRasiD10(g[5])
    shaniD10 = getRasiD10(g[6])
    rahuD10 = getRasiD10(g[7])
    ketuD10 = getRasiD10(g[7] + 180)
    lagnaD10 = getRasiD10(ascendant)

    suryaD16 = getRasiD16(g[0])
    chandraD16 = getRasiD16(g[1])
    mangalD16 = getRasiD16(g[2])
    budhD16 = getRasiD16(g[3])
    guruD16 = getRasiD16(g[4])
    shukraD16 = getRasiD16(g[5])
    shaniD16 = getRasiD16(g[6])
    rahuD16 = getRasiD16(g[7])
    ketuD16 = getRasiD16(g[7] + 180)
    lagnaD16 = getRasiD16(ascendant)

    suryaD27 = getRasiD27(g[0])
    chandraD27 = getRasiD27(g[1])
    mangalD27 = getRasiD27(g[2])
    budhD27 = getRasiD27(g[3])
    guruD27 = getRasiD27(g[4])
    shukraD27 = getRasiD27(g[5])
    shaniD27 = getRasiD27(g[6])
    rahuD27 = getRasiD27(g[7])
    ketuD27 = getRasiD27(g[7] + 180)
    lagnaD27 = getRasiD27(ascendant)

    suryaD30 = getRasiD30(g[0])
    chandraD30 = getRasiD30(g[1])
    mangalD30 = getRasiD30(g[2])
    budhD30 = getRasiD30(g[3])
    guruD30 = getRasiD30(g[4])
    shukraD30 = getRasiD30(g[5])
    shaniD30 = getRasiD30(g[6])
    rahuD30 = getRasiD30(g[7])
    ketuD30 = getRasiD30(g[7] + 180)
    lagnaD30 = getRasiD30(ascendant)

    surya = getRasi(g[0]) 
    suryab = Bhaava(g[0], grahas) 
    suryani = naks(g[0]) 
    suryan = naks(g[0])  
    chandra = getRasi(g[1])     
    chandrab = Bhaava(g[1], grahas) 
    chandrani = naks(g[1]) 
    chandran = naks(g[1])
    mangal = getRasi(g[2]) 
    mangalb = Bhaava(g[2], grahas) 
    mangalani = naks(g[2]) 
    mangalani = naks(g[2])
    budh = getRasi(g[3]) 
    budhb = Bhaava(g[3], grahas) 
    budhani = naks(g[3]) 
    budhan = naks(g[3])
    guru = getRasi(g[4])
    gurub = Bhaava(g[4], grahas) 
    guruani = naks(g[4]) 
    guruan = naks(g[4])
    shukra = getRasi(g[5]) 
    shukrab = Bhaava(g[5], grahas) 
    shukrani = naks(g[5]) 
    shukran = naks(g[5])
    shani = getRasi(g[6]) 
    shanib = Bhaava(g[6], grahas) 
    shaniani = naks(g[6]) 
    shanian = naks(g[6]) 
    shanine = ansh5(g[5])
    rahu = getRasi(g[7]) 
    rahub = Bhaava(g[7], grahas) 
    rahuan = naks(g[7]) 
    rahuni = naks(g[7]) 
    rahuanim = ansh5(g[7])
    ketu = getRasi(reddeg(g[7] + 180)) 
    ketub = Bhaava(reddeg(g[7] + 180), grahas) 
    ketuan = naks(g[8]) 
    ketuanim = ansh5(reddeg(g[7] + 180))
    lagna = getRasi(ascendant) 
    lagnani = naks(ascendant) 
    lagna = naks(ascendant)
    lagna = getRasi(ascendant)

    House1 = 0
    House2 = 1
    House3 = 2
    House4 = 3
    House5 = 4
    House6 = 5
    House7 = 6
    House8 = 7
    House9 = 8
    House10 = 9
    House11 = 10
    House12 = 11

    
    phala = [
        {'group':'Ascendant','rule':	lagnani==0	, 'des':'	स्वभाव	',    'sou':'Phala Deepika', 'result':'	जीवंतए सहजए सर्जकए बुद्धिमानए छोटा कदए स्पोर्टीए मजबूतए आकर्षकए करिश्माईए स्टाइलिशए भव्यए मासूम।	'},
        {'group':'Ascendant','rule':	lagna==0	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==1	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==2	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==3	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==4	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==5	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==6	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==7	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==8	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==9	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==10	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},
        {'group':'Ascendant','rule':	lagna==11	, 'des':'	𝙗𝙤𝙙𝙮	',    'sou':'Phala Deepika', 'result':'		 '},


        ]
    prediction_html = ''
    for i, ph in enumerate(phala, start=1):
        if ph.get('rule'):
            prediction_html += f'\n\n{ph["des"]} :- {ph["result"]}'
    
    prediction_html += ''
    return prediction_html

def red12(g):
    return g % 12 if g % 12 >= 0 else (g % 12) + 12

def get_nakshatra_for_planets(gr):
    nakshatra_names = [
        '1,2,3,5,10,20,24,28,30,46', '7,8,24,28,33,51', '7,14,21,30,42,43,47,79', '3,11,24,29,36,63,72,83,84', '4,8,17,25,34,46,50,65,73,79', '15,18,24,37,42,61', '30,31,39,40,66,70,84,86,89', 
        '16,24,33', '14,17,27,30,33,41,65', '18,31,46,49,51,52,60,61,70,89', '5,24,29,37', '9,18,35,45,62,70,79,89', '3,6,22,24,46,50,68,77,79', 
        '5,20,32,49,69,78,79', '3,20,30,37,39,41,45,55,84', '5,33,37,39,43,48,49,57,60,66,82,89', '25,26,29,27,37,44,48,63,66,78,84', '6,7,8,24,27,28,36', '6,7,25,30,35,34,36,60,64,81,87', '8,12,16,24,30,64,72,77,88', 
        '5,12,324,36,44,48,59,61,67,72,77,83', '5,15,33,35,41,50,55,57,58,77', '27,36,40,39,45', '24,28,35,42,51,53,64,65,67,68,75,81', '18,21,24,37,39,45,61,63,65,69,87', 
        '11,18,19,24,64,68,70,75,76', '5,6,12,32,42,60'
    ]
    
    nakshatra_degrees = 13.33  # Each nakshatra spans 13.33 degrees
    total_degrees = 360
    
    # Function to calculate nakshatra based on planet's degree
    def get_nakshatra(degree):
        normalized_degree = degree % total_degrees  # To handle cases where degree > 360
        nakshatra_index = int(normalized_degree // nakshatra_degrees)  # Determine the nakshatra index
        return nakshatra_names[nakshatra_index]
    
    # Initialize an empty list to store nakshatras
    nakshatra_results = []
    
    # Loop through each degree in the list and determine nakshatra
    for degree in gr:
        nakshatra_results.append(get_nakshatra(degree))  # Append nakshatra to the results list
    
    return nakshatra_results  # Return the list of nakshatras



@app.route('/api/calculate_sun_moon', methods=['POST'])
@cross_origin()
def calculate_sun_moon():
    data = request.get_json()
    name = data['name']
    date_of_birth = data['date_of_birth']
    time_of_birth = data['time_of_birth']
    place_of_birth = data['place_of_birth']
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])
    timezone_offset = data['timezone_offset']
    julian_date = data['julian_date']

    dob = datetime.strptime(date_of_birth + ' ' + time_of_birth, '%Y-%m-%d %H:%M')
    jday = julian_date
    obs = {'latitude': latitude, 'longitude': longitude}

    # Assuming calculateAscendant and getGrahas are defined somewhere
    ascendant, mc, bhaava_madya, bhaava_sandhi, ayanamsa, node = calculateAscendant(dob, jday, latitude, longitude, timezone_offset)
    
    g, g_next, speed = getGrahas(jday, longitude, latitude)
    
    gr = [0] * 10
    for i in range(9):
        if g[i] is not None and ayanamsa is not None:
            gr[i] = (float(g[i]) + float(ayanamsa) + 36000) % 360
        else:
            gr[i] = None
    
    if node is not None and ayanamsa is not None:
        gr[7] = (float(node) + float(ayanamsa) + 36000) % 360
        gr[8] = (float(node) + float(ayanamsa) + 180 + 36000) % 360
        gr[9] = (float(node) + 36000) % 360
    else:
        gr[7] = None
        gr[8] = None
        gr[9] = None
    
    T = (((360 + gr[1] - gr[0]) % 360) / 12)
    N = (gr[1] / (13 + 1 / 3))
    Y = (((gr[1] + gr[0]) % 360) / (13 + 1 / 3))
    K = (((360 + gr[1] - gr[0]) % 360) / 12) * 2
    R = (gr[1] / 30)

    Ti = ["S.Prathama", "S.Dwitiya", "S.Tritiya", "S.Chaturthi", "S.Panchami", "S.Shashti", "S.Saptami", "S.Ashtami", "S.Navami", "S.Dasami", "S.Ekadashi", "S.Dwadasi", "S.Trayodasi", "S.Chaturdashi", "Poornima", "K.Prathama", "K.Dwitiya", "K.Tritiya", "K.Chaturthi", "K.Panchami", "K.Shashti", "K.Saptami", "K.Ashtami", "K.Navami", "K.Dasami", "K.Ekadashi", "K.Dwadasi", "K.Trayodasi", "K.Chaturdashi", "Amavasya"]
    Na = ["Ashvini", "Bharani", "Kritika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya", "Ashlesha", "Magha", "P.Phalguni", "U.Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "P.Ashadha", "U.Ashadha", "Shravan", "Dhanistha", "Shatabhishaj", "P.Bhadrapad", "U.Bhadrapad", "Revati"]
    Yo = ["Vishkambha", "Priti", "Ayushman", "Saubhagya", "Shobhana", "Atiganda", "Sukarman", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva", "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Varigha", "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla", "Brahma", "Mahendra", "Vaidhriti"]
    Ka = ["Kimstug", "Bhava", "Bhaala", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Bhava", "Bhaalava", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Bhava", "Bhaalava", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Bhava", "Bhaalava", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Bhava", "Bhaalava", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Bhava", "Bhaalava", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Bhava", "Bhaalava", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Bhava", "Bhaalava", "Kaulava", "Taitula", "Garija", "Vanija", "Vishti", "Shakuni", "Chatushpada", "Naaga"]
    Ra = ["Mesh","Vrash","Mithun","kark","Singh","Kanya","tula","vraschik","dhanu","Makar","Kumbh","Meena"]

    tithi = Ti[int(T)]
    karan = Ka[int(K)]
    yog = Yo[int(Y)]
    nakshatra = Na[int(N)]
    raasi = Ra[int(R)]

    gr_degrees = [todeg(deg) for deg in gr if deg is not None]

    varga_chart = {
        'D1': vargas(1, gr),
        'D9': vargas(9, gr),
        # Add other Vargas as needed
    }
    prediction_html = pha(gr, ascendant)

    d = vargas(1, gr)
    e = 1  # Example chart type (Navamsa)
    s = vChart(d, e)  # This should work if `gr` is defined
    print(s)

    nakshatra_for_planets = get_nakshatra_for_planets(gr)
    print(nakshatra_for_planets)
    

    return jsonify({
        'name': name,
        'date_of_birth': date_of_birth,
        'time_of_birth': time_of_birth,
        'place_of_birth': place_of_birth,
        'latitude': latitude,
        'longitude': longitude,
        'ayanamsa': ayanamsa,
        'node': node,
        'ascendant': ascendant,
        'mc': mc,
        'bhaava_madya': bhaava_madya,
        'bhaava_sandhi': bhaava_sandhi,
        'gr': gr,
        'gr_degrees': gr_degrees,
        'tithi': tithi,
        'yog': yog,
        'karan': karan,
        'nakshatra': nakshatra,
        'raasi': raasi,
        'naks' : nakshatra_for_planets,
        'varga_chart': varga_chart,
        'prediction' : prediction_html,
        'varga_result': s,
    })

def reddeg(d):
    return (d + 360) % 360

def getRasi(d):
    return floor(reddeg(d) / 30)

def fra(t):
    return t - floor(t)

def tt(t, y=0):
    if y == 0:
        z = f"{t:02d}" if t < 10 else str(t)
    else:
        if t > 100:
            z = str(t)
        elif t > 10:
            z = f"{t:02d}"
        else:
            z = f"{t:03d}"
    return z

w = ["Ar", "Ta", "Ge", "Ca", "Le", "Vi", "Li", "Sc", "Sg", "Cp", "Aq", "Pi"]

def todeg(t):
    r = fra(t)
    a = floor(t)
    r2 = r * 60
    return f"{tt(a % 30)} {w[floor(a / 30)]} {tt(floor(r2))}' {tt(floor(fra(r2) * 60))}\""


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)