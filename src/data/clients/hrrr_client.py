# download subset of HRRR data based on 
#   - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
#   - fire location
#   - variables of interest
#       - hdw:              'vpd_2m', 'wind_speed'
#       - hwp: HWP=0.213*G^(1.5)*vpd^(0.73)(1-M)^(5.10)S
#           - https://doi.org/10.1175/WAF-D-24-0068.1
#           - G: max(3, 10-m wind gust potential)
#           - VPD: 2-m vapor pressure deficit
#               - calculate from relative humidity (RH) and temperature (TMP)
#               - SVP = 610.7*10^{7.5*TMP/(237.3+T)}/1000
#               - AVP = SVP * RH/100
#               - VPD = SVP(1 - RH/100) 
#           - M: soil moisture availability (MSTAV)
#           - S: snow water equivalent term (WEASD)
#       - hrrr_met: VPD, WIND

from herbie import Herbie

class HRRRClient:
    def __init__(self, date, lead_time):
        self.H = Herbie(date, model = "hrrr",  product="sfc", fxx=lead_time)


if __name__ == "__main__":
    hc = HRRRClient(
        date = "2021-07-01 12:00",
        lead_time=6)
    import ipdb; ipdb.set_trace()