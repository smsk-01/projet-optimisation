# N.B Les algorithmes de la question 3 de la partie 2 concernant la complexité temporelle et celui question 4.1 mettent un petit peu de temps à tourner.
# Le "run all" devrait prendre une minute. 

import numpy as np
from scipy import optimize
from casadi import *
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import random

#constantes du problème
a, b, L = 0, 1, 1.25
N = 40
d = np.full(N, (b-a)/N)


# # **1 Etude du problème d'optimisation**

# ## 1.1

# Le coût à minimiser s'écrit : 
# $
#     f : z\mapsto c^Tz\ avec,
#   z = 
#   \begin{pmatrix}
#     y_{0}\\
#     y_{1}\\
#     \vdots\\
#     y_{N}
#   \end{pmatrix}
#   \in \mathbb{R}^{N+1} \, ; \, c = 
#   \begin{pmatrix}
#     d_{0}\\
#     d_{0} + d_{1}\\
#     \vdots\\
#     d_{N-2} + d_{N-1}\\
#     d_{N-1}
#   \end{pmatrix}
#   \in \mathbb{R}^{N+1}
# $

# ## 1.2

# Il y a 3 contraintes :
# - l'ordonnée du premier point est nulle
# - l'ordonnée du dernier point est nulle
# - le périmètre de la clôture vaut L
#
# $\begin{array}{ccccc}
# g & : & \mathbb{R}^{N+1} & \to & \mathbb{R}^{3} \\
#  & & z & \mapsto & \begin{pmatrix}
#     z_{0}\\
#     z_{N}\\
#     -L + \sum_{i = 0}^{N-1} \sqrt{d_{i}^{2} + (z_{i+1}-z_{i})^{2}}\\
#   \end{pmatrix} \\
# \end{array}$

# ## 1.3

# La fonction coût est affine donc elle est convexe. Il en va de même pour les deux premières composantes de la fonction g.
# Montrons que la troisième composante de g l'est aussi.
# On note 
#
# $\begin{array}{ccccc}
# w & : & \mathbb{R}^{N+1} & \to & \mathbb{R} \\
#  & & z & \mapsto &     
#     -L + \sum_{i = 0}^{N-1} \sqrt{d_{i}^{2} + (z_{i+1}-z_{i})^{2}}\\
# \end{array}$
#
# $ \forall 1<k<N, \frac{\partial w}{\partial z_{k}} = \frac{z_{k}-z_{k-1}}{\sqrt{d_{k-1}^{2}+(z_{k}-z_{k-1})^{2}}} 
# - \frac{z_{k+1}-z_k}{\sqrt{d_{k}^{2}+(z_{k+1}-z_{k})^{2}}}$
#
# $ \frac{\partial w}{\partial z_{0}} = -\frac{z_1-z_0}{\sqrt{d_{0}^{2}+(z_{1}-z_{0})^{2}}}$
#
# $ \frac{\partial w}{\partial z_{N}} = \frac{z_N-z_{N-1}}{\sqrt{d_{N-1}^{2}+(z_{N}-z_{N-1})^{2}}}$
#
# $ \forall 1<k<N, \frac{\partial^2 f}{\partial z_k^2} = \frac{d_{k-1}^2}{(d_{k-1}^{2}+(z_{k}-z_{k-1})^{2})^{\frac{3}{2}}} 
# + \frac{d_{k}^{2}}{(d_{k}^{2}+(z_{k+1}-z_{k})^{2})^{\frac{3}{2}}} ~ ; ~ 
# \frac{\partial^2 f}{\partial z_{k-1} \partial z_k} = -\frac{d_{k-1}^2}{(d_{k-1}^{2}+(z_{k}-z_{k-1})^{2})^{\frac{3}{2}}} $
#
# $ \frac{\partial^2 f}{\partial z_0^2} = \frac{d_{0}^2}{(d_{0}^{2}+(z_{1}-z_{0})^{2})^{\frac{3}{2}}} $
#
# $ \frac{\partial^2 f}{\partial z_N^2} = \frac{d_{N-1}^2}{(d_{N-1}^{2}+(z_{N}-z_{N-1})^{2})^{\frac{3}{2}}} $
#
# $ 
#   H(w) = 
#   \begin{pmatrix}
#     a_{0} & -a_{0} & 0 & \cdots & 0 \\
#     -a_{0} & a_{0} + a_{1} & -a_{1} & \cdots & 0\\
#     0 & -a_{1} & a_{1} + a_{2} & \cdots & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots \\
#     0 & 0 & 0 & \cdots & a_{N}
#   \end{pmatrix}
#  ~ où ~ \forall 1\leq k\leq N, a_k = \frac{d_{k}^2}{(d_{k}^{2}+(z_{k+1}-z_{k})^{2})^{\frac{3}{2}}} $
#
# Or d'après un corollaire du lemme d'Hadamard :
# Soit $A \in M_n(\mathbb{C})$ une matrice hermitienne. Alors si A est à diagonale dominante, elle est positive si et seulement si ses coefficients diagonaux sont des réels positifs ou nuls. 
# On en déduit que H(w) est positive et que w est convexe.
# Chaque composante de g étant convexe, la fonction g est convexe.
#
# La fonction coût et la fonction g sont de classe $C^\infty$.

# # **2 Résolution numérique**

# ## 2.1

# On crée une fonction qui permet de générer le vecteur c ainsi que les 3 contraintes du problème.
def donnees(d, L):
    N = len(d)
    c = []
    c.append(d[0]/2)
    for i in range(N-1):
        c.append((d[i] + d[i+1])/2)
    c.append(d[N-1]/2)   
    g = lambda z : np.concatenate(((np.array([z[0]])), np.array([z[N]]),np.array([sum([(d[i]**2 + (z[i+1]-z[i])**2)**0.5 for i in range(N)])-L])), axis=0)
    c1 = lambda z : g(z)[0]
    c2 = lambda z : g(z)[1]
    c3 = lambda z : g(z)[2]
    return c, c1, c2, c3


def optimal_curve(a, b, L, N=10, d = None, init_guess = None):
    if d is None :
        d = np.full(N, (b-a)/N)
    if init_guess is None :
        z = np.zeros(N + 1)
        z[1] = np.sqrt(((L - np.sum(d) + d[0] + d[1])**2 / 4) - d[0]**2)
        init_guess = np.array(z)
    don = donnees(d, L)
    cout = lambda z : -np.dot(don[0], z)
    x_opt = np.cumsum(np.concatenate(([a], d)))
    d = np.concatenate((np.array([0]),d),axis=0)
    
    Constraints = [{'type' : 'eq', 'fun' : don[1]}, {'type' : 'eq', 'fun' : don[2]}, 
                      {'type' : 'eq', 'fun' : don[3]}, {'type' : 'ineq', 'fun' : lambda z: z}]

    y_opt = (optimize.minimize(cout, init_guess, method = 'SLSQP', constraints = Constraints)).x
    t_opt = (y_opt[1:] - y_opt[:-1]) / d[1:]
    return np.array(x_opt), y_opt, t_opt


tab = optimal_curve(a, b, 28000)


# +
def init_guess(a, b, L, N=10, d = None):
    if d is None :
        d = np.array([(b-a)/N for i in range(N)])
    z = [0]
    z.append(((L - np.sum(d) + d[0] + d[1])**2/4 - d[0]**2)**0.5)
    for i in range(N-1):
        z.append(0)
    return np.array(z)

plt.plot(tab[0], init_guess(a, b, 1.25))
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# -


# On peut choisir un init_guess avec un triangle lorsque celui-ci n'est pas donné.

# ## 2.2

plt.plot(tab[0], tab[1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

L_test = [optimal_curve(0, 1, 10**i, 40) for i in range(1,8)]    
for i,_ in enumerate(L_test) :
    plt.plot(L_test[i][0], L_test[i][1])
    plt.grid('True')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# La solution semble être convexe. Pour des grandes valeurs du périmètre, typiquement à partir de L = 28000 dans l'exemple ci-dessus, le champ délimité par l'enclos n'est plus convexe. Pour que la solution reste convexe, il ne faut donc pas que $ b-a \ll L $.

# ## 2.3

# +
def mean_time(N):
    list_t = []
    t0 = time.time()
    tab = optimal_curve(a, b, L, N)
    tf = time.time()
    return tf-t0

vec_time = np.vectorize(mean_time)

list_N = np.array([5*i for i in range(2,20)])
list_t = vec_time(list_N)

# -

std = np.std(list_t)
plt.errorbar(list_N, list_t, yerr=std, fmt='o-',capsize=4)
plt.plot(list_N, list_t)
plt.xlabel('N')
plt.ylabel("Temps d'execution")
plt.show()

# La complexité temporelle de l'algorithme ainsi implémenté semble être en $ O(N^{\alpha}) ~ avec ~ \alpha > 1.$ Essayons d'approcher $\alpha$ :

log_list_N = np.log(list_N)
log_list_t = np.log(list_t)
alpha, ordorign = np.polyfit(log_list_N, log_list_t, 1)
print(f"On estime la complexité temporelle en O({alpha})") #on trouve généralement un résultat proche de 2.1

# ## **3 Ajout d'une contrainte**

# ## 3.4

# $
# \begin{array}{ccccc}
# h & : & \mathbb{R}^{N+1} & \to & \mathbb{R}^{N+1} \\
#  & & z & \mapsto & \begin{pmatrix}
#     z_{0}-y_{max}(a)\\
#     z_{1}-y_{max}(a+d_{0})\\
#     \vdots\\
#     z_{i}-y_{max}(a+\sum_{k = 0}^{i-1} d_{i})\\
#     \vdots\\
#     z_{N}-y_{max}(a+\sum_{k = 0}^{N-1} d_{i})\\
#   \end{pmatrix} \\
# \end{array} \\ 
# $
# Qui se réécrit : $h : \mathbf{z} \mapsto A\mathbf{z} + \mathbf{b}$,
# Avec $A = I_{N+1}$ et $\mathbf{b} = (-y_{max}(a),-y_{max}(a+d_{0}), \dots,-y_{max}(a+\sum_{k = 0}^{N-1} d_{i})) \\$ 
# NB : on ajoute aussi une contrainte sur la positivité du vecteur $\mathbf{z}$ par précaution.
#

# ## 3.5

df = pd.read_csv('river_positions.csv')
df.head()

plt.xlabel("x")
plt.ylabel("y")
plt.title("rivière")
plt.plot(df['x'],df['y'])


def y_max_func(x, interpolator=None):
    eps = 1e-6
    x += eps
    if interpolator is None:
        interpolator = interp1d(df["x"], df["y"], kind='linear', assume_sorted=False, bounds_error=False, fill_value='extrapolate')
    return interpolator(x)
y_max_func_vect = np.vectorize(y_max_func)

# ## 3.6

h = lambda z,a,d,y_max_func_vect : y_max_func_vect(a + np.cumsum(d)) - 0.5 - z #on met l'enclos 50cm avant la rivière.
def optimal_bounded_curve(a, b, L, y_max_func_vect, N=10, d = None, init_guess = None):
    if d is None :
        d = np.full(N, (b-a)/N)
    if init_guess is None :
        z = np.zeros(N + 1)
        z[1] = np.sqrt(((L - np.sum(d) + d[0] + d[1])**2 / 4) - d[0]**2)
        init_guess = np.array(z)
    don = donnees(d, L)
    cout = lambda z : -np.dot(don[0], z)
    x_opt = np.cumsum(np.concatenate(([a], d)))
    d = np.concatenate((np.array([0]),d),axis=0)
    
    
    Constraints = [{'type' : 'ineq', 'fun' : lambda z : h(z,a,d,y_max_func_vect)}, {'type' : 'eq', 'fun' : don[1]}, {'type' : 'eq', 'fun' : don[2]}, 
                      {'type' : 'eq', 'fun' : don[3]}, {'type' : 'ineq', 'fun' : lambda z: z}]

    y_opt = (optimize.minimize(cout, init_guess, method = 'SLSQP', constraints = Constraints)).x
    t_opt = (y_opt[1:] - y_opt[:-1]) / d[1:]
    
    return np.array(x_opt), y_opt, t_opt

tab2 = optimal_bounded_curve(0, 150, 220, y_max_func_vect)


# +
def moutons(X, Y, nb_moutons) :
    moutons_x, moutons_y = [], []
    for j in range(nb_moutons) :
        i = random.randrange(len(X))
        moutons_x.append(X[i])
        moutons_y.append(Y[i]-np.random.uniform(0,Y[i]))
    return np.array(moutons_x), np.array(moutons_y)
    

moutons_x, moutons_y = moutons(tab2[0],tab2[1],20)
plt.plot(df['x'],df['y'])
plt.plot(tab2[0],tab2[1]) 
plt.scatter(moutons_x, moutons_y, marker='+')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Le champ du berger et ses moutons")
plt.legend(["rivière", "enclos", "moutons"])
plt.grid(True)
# -

# ## 4 Pour aller plus loin

# ## 4.1
# Nous allons désormais raffiner la distribution des pas di afin de maximiser la surface disponible. Nous travaillerons sur la courbe optimal_bounded_curve car cela permet de mieux voir le réel intérêt de cette nouvelle distribution.\
# Notre fonction à maximiser est l'aire sous la courbe "optimal_bounded_curve" en fonction de d.\
# Nous avons pour contrainte : 
#
# $
#     c : d\mapsto \sum_{i=0}^{N-1} d_{i} = {b-a} 
# $
#
# L'algorithme est assez long donc nous prendrons 10 points.
#

# +
import numpy as np
from scipy import optimize

def cost(d, a=0, b=150, L=250, y_max_func_vect=y_max_func_vect, N=10):
    x_opt, y_opt, t_opt = optimal_bounded_curve(a, b, L, y_max_func_vect, N, d)
    return -np.trapz(y_opt, x_opt)

def optimize_d_bounded(a, b, L, y_max_func_vect, N=10):
    bounds = [((b - a) / (N * 10), (b - a) / (N / 2)) for _ in range(N)]
    init_guess = np.full(N, (b - a) / N)
    constraint_d = lambda d : sum(d) -(b - a)
    Constraint = [{'type' : 'eq', 'fun' : constraint_d}]
    res = optimize.minimize(cost, init_guess, args=(a, b, L, y_max_func_vect, N), method="SLSQP", bounds=bounds, constraints = Constraint)
    d_opt = res.x
    x_opt, y_opt, t_opt = optimal_bounded_curve(a, b, L, y_max_func_vect, N, d_opt)

    return d_opt, x_opt, y_opt, t_opt


# -

d_opt, x_opt, y_opt, t_opt = optimize_d_bounded(0, 150, 250, y_max_func_vect, N=10)

x3, y3, t3 = optimal_bounded_curve(0, 150, 250, y_max_func_vect, N=10)
moutons_x, moutons_y = moutons(x_opt, y_opt,30)
plt.plot(df['x'],df['y'])
plt.plot(x3, y3) 
plt.plot(x_opt, y_opt)
plt.scatter(moutons_x, moutons_y, marker='+')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparaison entre un enclos avec une répartition des di uniforme et un enclos avec une répartition des di raffinée")
plt.legend(["rivière", "enclos d uniforme", "enclos d raffiné", "moutons"])
plt.grid(True)
print(f"La distance entre les points devra être {d_opt}")


# L'enclos vert maximise la surface disponible pour les moutons en affinant la disposition des di.

# ## 4.2

# Cette fois ci : $ z = 
#   \begin{pmatrix}
#     x_{0}\\
#     x_{1}\\
#     \vdots\\
#     x_{N} \\
#     y_{0}\\
#     y_{1}\\
#     \vdots\\
#     y_{N}
#   \end{pmatrix}
#   \in \mathbb{R}^{2N+2} $
#   
#   
# Le coût à minimiser s'écrit : 
# $
#     f : z\mapsto \sum_{i=0}^{N-1} \frac{(z_{i+1}-z_{i})(z_{N+2+i}+z_{N+1+i})}{2} 
# $
#
# Il faut ajouter comme contraintes : $ \forall 0 \leq i \leq N ~ , ~ a \leq x_{i} \leq b $

# + endofcell="--"
def donnees2(a, b, L, N):  
    g = lambda z : np.concatenate(((np.array([z[N+1]])), np.array([z[2*N+1]]),
                                   np.array([sum([((z[N+2+i]-z[N+1+i])**2 + (z[i+1]-z[i])**2)**0.5 for i in range(N)])-L]), 
                                   np.array([z[0]-a]), np.array([z[N]-b])), axis=0)
    c1 = lambda z : g(z)[0]
    c2 = lambda z : g(z)[1]
    c3 = lambda z : g(z)[2]
    c4 = lambda z : g(z)[3]
    c5 = lambda z : g(z)[4]
    return c1, c2, c3, c4, c5

def cout2(z):
    aire = 0
    N = len(z)//2-1
    for i in range(N):
        aire += 0.5*(z[i+1] - z[i])*(z[N+i+2] + z[N+i+1])
    return -aire


# -

def optimal_curve2(a, b, L, N=10, init_guess = None):
    if init_guess is None :
        z1 = np.zeros(N + 1)
        for i in range(N+1):
            z1[i] = a + (b-a)*i/N
        z2 = np.zeros(N + 1)
        z2[1] = np.sqrt(((L - np.sum(d) -z1[0] + z1[2])**2 / 4) - (z1[1]-z1[0])**2)
        init_guess = np.concatenate((z1, z2), axis = 0)
    don = donnees2(a, b, L, N)
    
    Constraints = [{'type' : 'eq', 'fun' : don[0]}, {'type' : 'eq', 'fun' : don[1]}, 
                      {'type' : 'eq', 'fun' : don[2]}, {'type' : 'eq', 'fun' : don[3]}, 
                   {'type' : 'eq', 'fun' : don[4]}, {'type' : 'ineq', 'fun' : lambda z : z[N+1:]}, 
                   {'type' : 'ineq', 'fun' : lambda z : z[:N+1] - a}, {'type' : 'ineq', 'fun' : lambda z : b-z[:N+1]}]
    
    

    z_opt = (optimize.minimize(cout2, init_guess, method = 'SLSQP', constraints = Constraints)).x
    x_opt = z_opt[:N+1]
    y_opt = z_opt[N+1:]
    return x_opt, y_opt

tab2 = optimal_curve2(a, b, 1.1)
tab3 = optimal_curve2(a, b, 3.5)
tab4 = optimal_curve2(a, b, 15)
# --

plt.plot(tab2[0], tab2[1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.plot(tab3[0], tab3[1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.plot(tab4[0], tab4[1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Une difficulté supplémentaire, lorsqu'on autorise les "retours en arrière", est de s'assurer que la clôture ne se recoupe pas sur elle-même.
