'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Author: Michaela Cully-Hugill

Description:
    This script computes the function in the condition for Theorem 1.4 of 
    "On the error term in the explicit formula of Riemann--von Mangoldt".
    It calculates the function for a fixed m and x_M to check whether we can 
    conclude that there are primes in the interval (n^m, (n+1)^m) for all
    positive x.

Inputs:
    - m: the power in the consecutive-powers interval (this value is coded as k in each function)
    - Range: can be set to 0 or 1; it corresponds to the index of the desired
    value of x_M in the list [10**3, 4*10**3], where x_M corresponds to the smallest x for which
    we can use a particular constant M in Theorem 1.2.

Additional notes:
    To get the result in Theorem 1.4, this script needs to be run for Range = 0 and Range = 1.
    The purpose is to check that the given m results in a positive condition function for
    10^3<x<4*10^3 and x>=4*10^3.
    
    The function G(y,k,a) uses the constants computed in Theorem 1.2 for 10^3 and 4*10^3.
    
    This script optimises over the parameter $\mu$ (coded here as 'a').

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import mpmath as mpm
from mpmath import mpf, sqrt, pi, log, exp, power, fdiv, findroot
from numpy import linspace

mpm.mp.dps = 100 #The number of decimal places of precision


#----------------------------- Constants -----------------------------------

c = 53.989 #Constant in the Vinogradov--Korobov zero-free region
R = 5.558691 #Constant in the classical zero-free region
A = 17.418 #Constant $C_1$ in the zero-density estimate, valid for sigma < 1
B = 5.272 #Constant $C_2$ in the zero-density estimate, valid for sigma >= 0.6

H0 = 3000175332800 #Constant in the partial verification of the Riemann hypothesis

s0 = 3/5 # The sigma at which the integral is split

xi = 2 #For the constant $\xi$ in Theorem 2.1


def h(k, X): # Interval length h in (x,x+h]
    return k*X**(1 - 1/k)

def x(y): # Variable x is parameterised as x = e^y
    return exp(y)

def W(a, y):
    return 2**(8/3)*exp(y-8*y*a/3)/(y*a-log(2))**2


#----------------Zero-free regions, note that these are used at 2T, and 2T=x^a=e^(ya)---------------

def ford(a,y):  #From Bellotti
    return fdiv(1,c*power(a*y,2/3)*power(log(a*y),1/3))

def fordclassical(a,y):  #From Hiary, Patel, and Yang
    def J(a,y):
        return 1/6*a*y + log(a*y) + log(0.618)

    def Rt(a,y):
        return (J(a,y) + 0.685 + 0.155*log(a*y))/(a*y*(0.04962-0.0196/(J(a,y)+1.15)))
    
    return 1/(Rt(a,y)*a*y)

def littlewood(a,y):  #From Yang
    return log(y*a)/(21.333*y*a)

def classical(a,y):  #From Mossinghoff, Trudgian, and Yang
    return 1/(R*y*a)

def v(a, y):
    return max(ford(a, y), littlewood(a, y), fordclassical(a,y), classical(a,y))


#---------------------------------- Functions --------------------------------------------
#Ms is a list containing the constants $M$ from Theorem 1.2 for log x > x_M, where each value
#corresponds to an x_M in [10**3, 4*10**3]. By Table 1, this also fixes the values of $\alpha$ and $\omega$.
Ms = [6.391, 5.462]

#In Theorem 1.2, $1-\omega$ is the power of log in the error term M*(x/T)*(log x)^(1-\omega)
#The list ps contains the values of $1-\omega$ corresponding to the previous values of M. 
ps = [1/10, 1/10]


def G(y, k, a): #Is the term in the condition function with the factor $G(x,h)$
    M = Ms[Range] 
    p = ps[Range]
    return xi*M*fdiv((x(y) + h(k, x(y)))*power(log(x(y) + h(k, x(y))),p) + x(y)*power(log(x(y)),p), power(x(y),a)*h(k, x(y)))

#Is the condition function $1 - F(x) - 2M*G(x,h)/(h*x^\mu)$
def R2(y, k, a, s):
    return 1 - (a/pi)*y*x(y)**(a + s - 1) - 2*A*a**3*y**4*(W(a, y)**(-v(a, y)) - W(a, y)**(s - 1))/log(W(a, y)) - 2*B*a**2*y**2*x(y)**(-v(a, y)) - G(y, k, a)


#------------Finds the values of T at which one ZF region is better than the other------------

log_root = findroot(lambda t: ford(1,t) - littlewood(1,t), 5*10**5, solver='secant') 
zf_switch1 = log_root #actual root is e^zf_switch
print('V--K zf-region better than Littlewood at log(T)=', round(zf_switch1,1))

log_root = findroot(lambda t: littlewood(1,t) - fordclassical(1,t), 2, solver='secant') 
zf_switch2 = log_root #actual root is e^zf_switch
print('Littlewood zf-region better than classical at log(T)=', round(zf_switch2,1))


#---------------------Power and range----------------------------
m = 90 #power

Range = 1 #Corresponds to the index of the list x_M = [10^3, 4*10^3]


#---------------------Range of x that the Bertrand interval covers--------------------------
#Constants are taken from Cully-Hugill & Lee (2024)

k1 = (39097/10000)*10**7 #For x0=4*10^18
k2 = (251949/100000)*10**11 #For x0=e^600

def Lx1(d,m):
    return m*log(m) + m*log(d - 1)
print('Bertrand interval covers the', m,'th powers interval from 4*10^(18) to e^',round(Lx1(k1,m),3)) 
print('Bertrand interval covers the', m,'th powers interval from e^600 to e^',round(Lx1(k2,m),3)) 


#---------------------------------------------------------------------------------------------
# The below section optimises over $\mu$ (coded as a) to find the smallest y>10**3 for which R2>0
#---------------------------------------------------------------------------------------------

#-----------------Converts T to x according to T = x^a, and returns y in x=e^y--------
def ySwt(a,switch):
    return (1/a)*(log(2)+switch)

#------------Calculates the range of admissible a---------------
a = round(1/m + 10**(-8),8)  #lower guess for mu

while R2(ySwt(a, zf_switch1), m, a, s0) <= 0: 
    a += 10**(-8)
    #print(a,R2(ySwt(a, zf_switch1), m, a, s0))
aLower = a

while R2(ySwt(a, zf_switch1), m, a, s0) > 0:
    a += 10**(-6)
    #print(a,R2(ySwt(a, zf_switch1), m, a, s0))
aUpper = a - 10**(-6)


#-----------------Generates range of possible values for a ($\mu$)------------------
aRange = linspace(aLower, aUpper, 50) #Uses previous section of code
#print(aRange)


#-----------------Finds the smallest y for which R2>0 for each a--------------------
smallest_a_y = [aRange[0],10**10]  #Initialises the pair [a,y] where y must be a value (ideally the smallest) for which R2>0, given a
order = [10**3, 4*10**3]

# if Range == 3: #For x\geq 10^5
#     for a in aRange:
#         y = 10**5
#         if R2(y, m, a, s0) > 0 and R2(y, m, a, s0) > R2(y, m, smallest_a_y[0], s0):
#             smallest_a_y = [a,y]
#         else:
#             while R2(y, m, a, s0) < 0 and y < smallest_a_y[1]:
#                 y += 10**3
#             if R2(y, m, a, s0) > 0 and y < smallest_a_y[1]:
#                 print(y)
#     final_a, final_y = smallest_a_y[0], smallest_a_y[1]
#     print('Condition >0 for ', final_y, '<= log(x) <=', round(ySwt(final_a, zf_switch1)), 'with a =', final_a)

y0 = 10**4 #The largest y we need to check the result holds for

for a in aRange:
    
    y = y0 
    if R2(y, m, a, s0) > 0 and R2(order[Range], m, a, s0) > 0:
        while R2(y-100, m, a, s0) > 0 and y-100 >= order[Range]:
            y -= 100
        if smallest_a_y[1] > y:
            print(y)
            smallest_a_y = [a,y]
    else:
        None #print(a, 'not viable')
final_a, final_y = smallest_a_y[0], smallest_a_y[1]

if final_y >y0:
    print('m=',m,'is not possible')
else:
    print('Condition function >0 for', final_y, '<= log(x) <=', y0, 'with a =', round(final_a,10))




#-------------------------OPTIONAL: Adjusting a to cover smaller y-----------------------
# ayPairs = [[final_a, [final_y, round(ySwt(final_a, zf_switch1))]]] # Will become a nested list of a values with the range of y it is used to confirm R2>0

# a1 = final_a
# y1 = final_y

# def options(a1,y1):
#     return [R2(y1, m, a1-10**(-6), s0), R2(y1, m, a1+10**(-6), s0)]

# while R2(y1, m, a1, s0) < max(options(a1,y1)) and y1 > 2500:
#     if options(a1,y1).index(max(options(a1,y1))) == 1:
#         a1 += 10**(-6)
#     else:
#         a1 -= 10**(-6)
#     while R2(y1-1, m, a1, s0) > 0:
#         y1 -= 1
# ayPairs.append([a1,[y1,final_y]])

# Below line shows that we cannot change a to get the next smallest integer y
#print(round(R2(y1-1, m, a1-10**(-7), s0),7), round(R2(y1-1, m, a1, s0),7), round(R2(y1-1, m, a1+10**(-7), s0),7))

#print(ayPairs)
#print('The switch-points are', ySwt(a1, zf_switch1))


#----------------------------- Plotting -------------------------------
#This plot is intended to display where the condition function is smallest for all x>x_M.
 
from mpmath import plot
alph = final_a
plot(lambda y: R2(y, m, alph, s0), [order[Range], ySwt(alph, zf_switch1)])
plot(lambda y: R2(y, m, alph, s0), [3*10**3, 5*10**3])
# plot([lambda y: classical(alph, y), lambda y: fordclassical(alph, y), lambda y: littlewood(alph, y), lambda y: ford(alph, y)], [100,10000])

