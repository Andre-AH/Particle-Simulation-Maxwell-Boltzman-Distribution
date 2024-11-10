import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.animation import FuncAnimation

X, Y = 0, 1
class MDSimulação:

    def __init__(self, pos, vel, r, m):
        """
        Inicializa a simulação com partículas esféricas com o mesmo raio
         r e massa m. As matrizes de estado: n x 2 pos e vel mantêm as n partículas e posições nas suas linhas como (x_i, y_i) e (vx_i, vy_i)

        """

        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.n = self.pos.shape[0]
        self.r = r
        self.m = m
        self.nsteps = 0

    def avanço(self, dt):
        """Avança a simulação por dt segundos."""

        self.nsteps += 1
        # Atualiza as posições das partículas de acordo com suas velocidades
        self.pos += self.vel * dt
        # Informações para colisões
        dist = squareform(pdist(self.pos))
        iarr, jarr = np.where(dist < 2 * self.r)
        k = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        # Para cada colisão atualiza a velocidade
        for i, j in zip(iarr, jarr):
            pos_i, vel_i = self.pos[i], self.vel[i]
            pos_j, vel_j =  self.pos[j], self.vel[j]
            rel_pos, rel_vel = pos_i - pos_j, vel_i - vel_j
            r_rel = np.matmul(rel_pos, rel_pos) # Velocidade relativa
            v_rel = np.matmul(rel_vel, rel_pos) # Produto vetorial entre Posição relativa e Velocidade relativa
            v_rel = 2 * rel_pos * v_rel / r_rel - rel_vel  # Colisão elástica (e linhas seguintes)
            v_cm = (vel_i + vel_j) / 2
            self.vel[i] = v_cm - v_rel/2
            self.vel[j] = v_cm + v_rel/2

        # Ressaltar as partículas que colidem com a parede e, quando necessário, mudar o vetor velocidade
        hit_left_wall = self.pos[:, X] < self.r
        hit_right_wall = self.pos[:, X] > 1 - self.r
        hit_bottom_wall = self.pos[:, Y] < self.r
        hit_top_wall = self.pos[:, Y] > 1 - self.r
        self.vel[hit_left_wall | hit_right_wall, X] *= -1
        self.vel[hit_bottom_wall | hit_top_wall, Y] *= -1

# -----------------------------------------------------------------------------------------------------------
# Input das propriedades:

# Número de partículas
n = int(input('Número de partículas: '))
# Introduzir escala para a distância. A dimensão da caixa é: 1/rscale.
rscale = 5.e6
# Usar raio de van der Waals da partícula:
gas = ['O', 'C', 'H', 'Ar']
ball_radius = [152*(10**(-10)), 170*(10**(-12)), 120*(10**(-12)), 188*(10**(-12))] #em nm
massa_molar = [0.016, 0.012, 0.001, 0.04]
tipo_gas = str(input('Gás [O/C/H/Ar]: '))
r_p = ball_radius[gas.index(tipo_gas)]
r = r_p * rscale
# Também aplicar uma escala para o tempo
tscale = 1e9    # i.e. tempo em nanosegundos.
# Usar a rapidez média como a velocidade média quadrática:
T = int(input('Temperatura [K]: '))
M = massa_molar[gas.index(tipo_gas)]
vrms = (3*8.3145*T/M)**(1/2) # vrms-velocidade média quadrática / R=8.3145
sbar = vrms * rscale / tscale # rapidez média
# Tempo e escala temporal.
FPS = int(input('FPS entre [10, 60] [recomendado 30]: '))
dt = 1/FPS
m = 1

# Inicializar as posições de forma randómica
pos = np.random.random((n, 2))
# Inicializar as velocidades das partículas com orientações e magnitudes aleatórias por volta de sbar
theta = np.random.random(n) * 2 * np.pi
s0 = sbar * np.random.random(n)
vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T

sim = MDSimulação(pos, vel, r, m)

# Preparar a figura
DPI = 100
width, height = 1000, 500
fig = plt.figure(figsize=(width/DPI, height/DPI), dpi=DPI)
fig.subplots_adjust(left=0, right=0.97)
sim_ax = fig.add_subplot(121, aspect='equal', autoscale_on=False)
sim_ax.set_xticks([])
sim_ax.set_yticks([])
# Aumentar a espessura da caixa
for spine in sim_ax.spines.values():
    spine.set_linewidth(2)

speed_ax = fig.add_subplot(122)
speed_ax.set_xlabel('Speed $v\,/m\,s^{-1}$')
speed_ax.set_ylabel('$f(v)$')

particles, = sim_ax.plot([], [], 'bo')

class Histogram:
    """Uma classe para desenhar um histograma"""

    def __init__(self, data, xmax, nbars, density=False):
        """Inicializa o histograma a partir dos dados e bins."""
        self.nbars = nbars
        self.density = density
        self.bins = np.linspace(0, xmax, nbars)
        self.hist, bins = np.histogram(data, self.bins, density=density)

        self.left = np.array(bins[:-1])
        self.right = np.array(bins[1:])
        self.bottom = np.zeros(len(self.left))
        self.top = self.bottom + self.hist
        nrects = len(self.left)
        self.nverts = nrects * 5
        self.verts = np.zeros((self.nverts, 2))
        self.verts[0::5, 0] = self.left
        self.verts[0::5, 1] = self.bottom
        self.verts[1::5, 0] = self.left
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 0] = self.right
        self.verts[2::5, 1] = self.top
        self.verts[3::5, 0] = self.right
        self.verts[3::5, 1] = self.bottom

    def draw(self, ax):
        codes = np.ones(self.nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        barpath = path.Path(self.verts, codes)
        self.patch = patches.PathPatch(barpath, fc='tab:green', ec='k',
                                  lw=0.5, alpha=0.5)
        ax.add_patch(self.patch)

    def update(self, data):
        self.hist, bins = np.histogram(data, self.bins, density=self.density)
        self.top = self.bottom + self.hist
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 1] = self.top


def get_speeds(vel):
    """Retorna a magnitude da matriz (n,2) das velocidades vel."""
    return np.hypot(vel[:, X], vel[:, Y])

def get_KE(speeds):
    """Retorna a Energia Cinética total de todas as partículas (unidades com escala)."""
    return 0.5 * sim.m * sum(speeds**2)

speeds = get_speeds(sim.vel)
speed_hist = Histogram(speeds, 2 * sbar, 60, density=True)
speed_hist.draw(speed_ax)
speed_ax.set_xlim(speed_hist.left[0], speed_hist.right[-1])
ticks = np.linspace(0, 2000, 7, dtype=int)
speed_ax.set_xticks(ticks * rscale/tscale)
speed_ax.set_xticklabels([str(tick) for tick in ticks])
speed_ax.set_yticks([])

fig.tight_layout()

# Distribuição de Maxwell-Boltzman para as velocidades em 2D
mean_KE = get_KE(speeds) / n
a = sim.m / 2 / mean_KE
# Usar uma grade de alta resolução de pontos de velocidade para que a distribuição exata seja suave
sgrid_hi = np.linspace(0, speed_hist.bins[-1], 200)
f = 2 * a * sgrid_hi * np.exp(-a * sgrid_hi**2)
mb_line, = speed_ax.plot(sgrid_hi, f, c='0.7')
# Valor máximo da distribuição de Maxwell-Boltzmann.
fmax = np.sqrt(sim.m / mean_KE / np.e)
speed_ax.set_ylim(0, fmax)

sgrid = (speed_hist.bins[1:] + speed_hist.bins[:-1]) / 2
mb_est_line, = speed_ax.plot([], [], c='r')
mb_est = np.zeros(len(sgrid))

# Um rótulo de texto indicando o tempo e o número de cada frame de animação.
xlabel, ylabel = sgrid[-1] / 2, 0.8 * fmax
label = speed_ax.text(xlabel, ylabel, '$t$ = {:.1f}s, step = {:d}'.format(0, 0),backgroundcolor='w')

def init_anim():
    """Inicia a animação"""
    particles.set_data([], [])

    return particles, speed_hist.patch, mb_est_line, label

def animate(i):
    """Avança a animação"""
    global sim, verts, mb_est_line, mb_est
    sim.avanço(dt)

    particles.set_data(sim.pos[:, X], sim.pos[:, Y])
    particles.set_markersize(0.5)

    speeds = get_speeds(sim.vel)
    speed_hist.update(speeds)

    # Assim que a simulação se aproximar um pouco do equilíbrio, começar a calcular a distribuição média de velocidades para observar a aproximação à Distribuição de Maxwell-Boltzmann.
    if i >= IAV_START:
        mb_est += (speed_hist.hist - mb_est) / (i - IAV_START + 1)
        mb_est_line.set_data(sgrid, mb_est)

    label.set_text('$t$ = {:.1f} ns, step = {:d}'.format(i*dt, i))

    return particles, speed_hist.patch, mb_est_line, label

# Calcular a média da distribuição de velocidades apenas após o frame IAV_ST.
IAV_START = 200
# Número de frames
frames = 2000
anim = FuncAnimation(fig, animate, frames=frames, interval=10, blit=False,
                    init_func=init_anim)

plt.show()
print('Se necessário ajustar a escala do eixo x na linha 172')