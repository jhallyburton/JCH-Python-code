''' -*- Mode: Python; tab-width: 4; python-indent: 4; indent-tabs-mode: nil; -*- '''

# Copyright (c) 2023, John Hallyburton
#
# License: MIT license. See:
#  https://en.wikipedia.org/wiki/MIT_License
#
# Credits: thanks to many Wikipedia contributors to the following pages:
#  https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#
#  https://en.wikipedia.org/wiki/Greeks_(finance)
#
# For testing, try one of these:
#  https://goodcalculators.com/black-scholes-calculator/
#  https://www.omnicalculator.com/finance/black-scholes
#  https://michaelryanmoney.com/black-scholes-calculator/
#  https://blackscholes.io/black-scholes-calculator
#
from math import log, sqrt, pi, exp
import statistics
norm      = statistics.NormalDist(0.0, 1.0)
days_year = 365.256363

"""
s = current stock price
k = option strike price
t = time to option expiration (in calendar years)
r = risk-free interest rate, per year. e.g., 5.08% is 0.0508
v = volatility of the asset (per year), e.g., 14% is entered as 0.14
q = dividend rate (yield) per year, e.g., 3% is 0.03. Optional, default 0.0
and
norm.pdf = PDF of the standard normal distribution (mu=0, sigma=1)
norm.cdf = CDF of the standard normal distribution
I was thinking of using "phi" for the PDF and "PHI" for the CDF but decided
not to because it seemed a bit more confusing than spelling out PDF and CDF.

For r, the risk-free interest rate, Yahoo ^IRX, the 91-day T-Bill rate is helpful.
For longer options consider https://ycharts.com/indicators/1_year_treasury_rate
which of course supplies the one-year T-Bill interest rate.

Volatility can be calculated from past price data. See, for example,
https://www.fool.com/investing/how-to-invest/stocks/how-to-calculate-stock-volatility/
for a spreadsheet algorithm. Also,
https://www.investopedia.com/terms/v/volatility.asp
gives a good overview of "volatility". Note that option volatility varies somewhat
from historical volatility as calculated by the spreadsheet algorithm.
"""

# Fundamental equations:

def d1(s,k,t,r,v,q=0.0):
    return (log(s/k) + (r - q + v*v/2.0)*t) / (v*sqrt(t))

def d2(s,k,t,r,v,q=0.0):
    return (log(s/k) + (r - q - v*v/2.0)*t) / (v*sqrt(t))

def bs_call(s,k,t,r,v,q=0.0):
    "Black-Scholes model for European call options, allowing dividends."
    return s*exp(-q*t)*norm.cdf(d1(s,k,t,r,v,q))  - k*exp(-r*t)*norm.cdf(d2(s,k,t,r,v,q))

def bs_put(s,k,t,r,v,q=0.0):
    "Black-Scholes model for European put options, allowing dividends."
    return k*exp(-r*t)*norm.cdf(-d2(s,k,t,r,v,q)) - s*exp(-q*t)*norm.cdf(-d1(s,k,t,r,v,q))

# Greeks. Not all of these have been tested.

def call_delta(s,k,t,r,v,q=0.0):
    return exp(-q*t)*norm.cdf(d1(s,k,t,r,v,q))

def  put_delta(s,k,t,r,v,q=0.0):
    return -exp(-q*t)*norm.cdf(-d1(s,k,t,r,v,q))

def call_vega(s,k,t,r,v,q=0.0):
    return exp(-q*t)*s*norm.pdf(d1(s,k,t,r,v,q))*sqrt(t)

def  put_vega(s,k,t,r,v,q=0.0):
    return exp(-q*t)*s*norm.pdf(d1(s,k,t,r,v,q))*sqrt(t)

def call_theta(s,k,t,r,v,q=0.0):
    return -(exp(-q*t)*s*norm.pdf(d1(s,k,t,r,v,q))*v)/(2*sqrt(t)) - \
         r*k*exp(-r*t)*norm.cdf( d2(s,k,t,r,v,q)) + q*s*exp(-q*t)*norm.cdf( d1(s,k,t,r,v,q))

def  put_theta(s,k,t,r,v,q=0.0):
    return -(exp(-q*t)*s*norm.pdf(d1(s,k,t,r,v,q))*v)/(2*sqrt(t)) + \
         r*k*exp(-r*t)*norm.cdf(-d2(s,k,t,r,v,q)) - q*s*exp(-q*t)*norm.cdf(-d1(s,k,t,r,v,q))

def call_rho(s,k,t,r,v,q=0.0):
    return  k*t*exp(-r*t) * norm.cdf( d2(s,k,t,r,v,q))

def  put_rho(s,k,t,r,v,q=0.0):
    return -k*t*exp(-r*t) * norm.cdf(-d2(s,k,t,r,v,q))

def call_epsilon(s,k,t,r,v,q=0.0):
    return -s*t*exp(-q*t) * norm.cdf( d1(s,k,t,r,v,q))

def  put_epsilon(s,k,t,r,v,q=0.0):
    return  s*t*exp(-q*t) * norm.cdf(-d1(s,k,t,r,v,q))

def call_lambda(s,k,t,r,v,q=0.0):
    return call_delta(s,k,t,r,v,q) * s / bs_call(s,k,t,r,v,q)

def  put_lambda(s,k,t,r,v,q=0.0):
    return  put_delta(s,k,t,r,v,q) * s / bs_put (s,k,t,r,v,q)

#

def call_gamma(s,k,t,r,v,q=0.0):
    return exp(-q*t) * norm.pdf(d1(s,k,t,r,v,q)) / (s*v*sqrt(t))

def  put_gamma(s,k,t,r,v,q=0.0):
    return call_gamma(s,k,t,r,v,q)

def call_vanna(s,k,t,r,v,q=0.0):
    return -exp(-q*t) * norm.pdf(d1(s,k,t,r,v,q)) * d2(s,k,t,r,v,q) / v

def  put_vanna(s,k,t,r,v,q=0.0):
    return call_vanna(s,k,t,r,v,q)

def call_charm(s,k,t,r,v,q=0.0):
    _d1_ = d1(s,k,t,r,v,q)
    return  q*exp(-q*t) * norm.cdf( _d1_) - exp(-q*t) * norm.pdf(_d1_) *\
           ( (2.0*(r-q)*t - d2(s,k,t,r,v,q)*v*sqrt(t) ) / (2.0*t*v*sqrt(t)))

def  put_charm(s,k,t,r,v,q=0.0):
    _d1_ = d1(s,k,t,r,v,q)
    return -q*exp(-q*t) * norm.cdf(-_d1_) - exp(-q*t) * norm.pdf(_d1_) *\
           ( (2.0*(r-q)*t - d2(s,k,t,r,v,q)*v*sqrt(t) ) / (2.0*t*v*sqrt(t)) )

def call_vomma(s,k,t,r,v,q=0.0):
    _d1_ = d1(s,k,t,r,v,q)
    return s*exp(-q*t)*norm.pdf(_d1_) * sqrt(t) * (_d1_ * d2(s,k,t,r,v,q) / v)

def  put_vomma(s,k,t,r,v,q=0.0):
    return call_vomma(s,k,t,r,v,q)

def call_veta(s,k,t,r,v,q=0.0):
    _d1_ = d1(s,k,t,r,v,q)
    # Extra parens inserted to try to bracket terms
    return -s*exp(-q*t)*norm.pdf(_d1_)*sqrt(t) * \
        (q + ((r-q) * _d1_ / (v*sqrt(t))) - \
               ( (1.0 + _d1_ * d2(s,k,t,r,v,q)) / (2.0*t)) )

def  put_veta(s,k,t,r,v,q=0.0):
    return call_veta(s,k,t,r,v,q)
# 

def call_psi(s,k,t,r,v,q=0.0):
    vv = v * v                      # sigma^2 appears several times
    bp = (r-q) - vv/2.0             # big parens (see Wiki Greeks equations)
    sb = log(k/s) - bp * t          # square brackets
    cb = -(1.0/(2.0*vv*t)) * sb*sb  # curly brackets
    return (exp(-r*t) / (k * sqrt(2.0*pi*vv*t))) * exp(cb)

def  put_psi(s,k,t,r,v,q=0.0):
    return call_psi(s,k,t,r,v,q)

def call_speed(s,k,t,r,v,q=0.0):
    return -(call_gamma(s,k,t,r,v,q)/s) * (d1(s,k,t,r,v,q)/(v*sqrt(t)) + 1.0)

def  put_speed(s,k,t,r,v,q=0.0):
    return -( put_gamma(s,k,t,r,v,q)/s) * (d1(s,k,t,r,v,q)/(v*sqrt(t)) + 1.0)

def call_zomma(s,k,t,r,v,q=0.0):
    return call_gamma(s,k,t,r,v,q) * (d1(s,k,t,r,v,q) * d2(s,k,t,r,v,q) - 1.0) / v

def  put_zomma(s,k,t,r,v,q=0.0):
    return  put_gamma(s,k,t,r,v,q) * (d1(s,k,t,r,v,q) * d2(s,k,t,r,v,q) - 1.0) / v

def call_color(s,k,t,r,v,q=0.0):
    _d1_ = d1(s,k,t,r,v,q)
    sb = 2.0*q*t + 1.0 + \
        _d1_ * (2.0*(r-q)*t - d2(s,k,t,r,v,q)*v*sqrt(t)) / (v*sqrt(t))
    return -exp(-q*t)*(norm.pdf(_d1_)/(2.0*s*t*v*sqrt(t)) ) * sb

def  put_color(s,k,t,r,v,q=0.0):
    return call_color(s,k,t,r,v,q)

def call_ultima(s,k,t,r,v,q=0.0):
    d1d2 = d1(s,k,t,r,v,q) * d2(s,k,t,r,v,q)
    d1d1 = d1(s,k,t,r,v,q) * d1(s,k,t,r,v,q)
    d2d2 = d2(s,k,t,r,v,q) * d2(s,k,t,r,v,q)
    return - (call_vega(s,k,t,r,v,q) / (v*v)) * ( d1d2 * (1.0 - d1d2) + d1d1 + d2d2 )

def  put_ultima(s,k,t,r,v,q=0.0):
    d1d2 = d1(s,k,t,r,v,q) * d2(s,k,t,r,v,q)
    d1d1 = d1(s,k,t,r,v,q) * d1(s,k,t,r,v,q)
    d2d2 = d2(s,k,t,r,v,q) * d2(s,k,t,r,v,q)
    return - ( put_vega(s,k,t,r,v,q) / (v*v)) * ( d1d2 * (1.0 - d1d2) + d1d1 + d2d2 )

def call_dual_delta(s,k,t,r,v,q=0.0):
    return -exp(-r*t)*norm.cdf( d2(s,k,t,r,v,q))

def  put_dual_delta(s,k,t,r,v,q=0.0):
    return  exp(-r*t)*norm.cdf(-d2(s,k,t,r,v,q))

def call_dual_gamma(s,k,t,r,v,q=0.0):
    return exp(-r*t)*norm.pdf(d2(s,k,t,r,v,q)) / (k*v*sqrt(t))

def  put_dual_gamma(s,k,t,r,v,q=0.0):
    return call_dual_gamma(s,k,t,r,v,q)

# ------------------------- ------------------------- -------------------------
#
#                          Implied Volatility routines.
#
# There exist faster ways to calculate IV but they require external C/C++ code
# and it is not clear the trouble is worth the gain. These routines run fast
# enough, < 46usec/call on my 2.80 GHz Core i7-1165G7 laptop, that there is no
# perceptible delay in interactive use.
#
# Implied volatility routines calculate option volatility based on a known
# option price, plus the other options calculation parameters excluding
# volatility.
#
def call_iv(call_option_price, s, k, t, r, q=0.0):
    "Binary search iv with huge endpoints, ~25 iterations to convergence."
    vlo = 0.0001
    vhi = 10000.0
    while vhi-vlo > 0.00001:
        v = (vlo+vhi) * 0.50

        delta_p = call_option_price - bs_call(s,k,t,r,v,q)
        if abs(delta_p) < 0.0001:
            return v

        if delta_p > 0:
            vlo = v
        else:
            vhi = v

    return v # Close enuf

def put_iv(put_option_price, s, k, t, r, q=0.0):
    "Binary search iv with huge endpoints, ~25 iterations to convergence."
    vlo = 0.0001
    vhi = 10000.0
    while vhi-vlo > 0.00001:
        v = (vlo+vhi) * 0.50

        delta_p = put_option_price - bs_put(s,k,t,r,v,q)
        if abs(delta_p) < 0.0001:
            return v

        if delta_p > 0:
            vlo = v
        else:
            vhi = v

    return v # Close enuf

# Small test routine, more for timing than testing. Expect iv=0.1012
# Try testing with options pricing calculator (URLs at top of this file),
# entering 10.12% volatility and expecting call option price to come up 5.24.
if __name__ == "__main__":
    import time

    n  = 1_000_000     # One million iterations (< 1 min. on my laptop)
    t  = 36./days_year # time to expiration (years)
    
    t0 = time.time()
    for i in range(n): iv = call_iv(5.24, 437.18, 440.0, t, 0.0508)
    t1 = time.time()

    print("Time required for %d IV calls: %.3f seconds = %.3f microsec/IV (iv=%.4f)."\
          % (n,t1-t0,1_000_000*(t1-t0)/n,iv))
    exit(0)
