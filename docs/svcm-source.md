# SVCM source transcription — Киселёв, doctoral dissertation

This document transcribes the mathematical passages of
`Киселёв_докторская_v1.pdf` (105 pp., Russian) that the earlier text-extraction
pass silently dropped: the original `.docx` carries every formula as a legacy
WMF equation image, so `pdftotext`/`docx`-to-text extraction returns prose with
holes where the math was ("with functional ⟨gap⟩ and dynamics ⟨gap⟩"). Every
formula below was read directly off the rendered PDF page image, not
reconstructed from surrounding text.

**Source file:** `Киселёв_докторская_v1.pdf`
**Pages covered, in full detail:**
- pp. 48–62 — §1.3.1–1.3.3: the Pontryagin–Hamilton control problem, the
  ε‑optimality definition and the set $U_\varepsilon$, the SVCM method
  statement, Theorem 2 (its two conditions of applicability and its
  controllable/uncontrollable dichotomy), and the related-work comparison.
- pp. 74–84 — Chapter 2 continuation: the Kalman-filter/LQG projection, the
  MPC↔PMP cost-matrix correspondence, nominal vs. crisis scenario matrices,
  **Algorithm 1 ("Синтез сценариев поведения агента")** in full, and the start
  of §2.2 (spatial-knowledge augmentation / online parameter identification).

**Pages skimmed for symbol definitions used above** (not transcribed in full,
only formulas that define notation reused on pp. 48–62 / 74–84):
- pp. 63–73 — Chapter 2 opening: MDP value functions, actor-critic, and the
  MPC actor formulation that pp. 74–84 build on directly.

Transcribed 2026-08-15. Russian prose is kept verbatim for the substantive
passages; each block carries an English gloss and a page citation. Every
formula is reproduced as rendered; nothing here is inferred or reconstructed
from prose alone.

## Table of contents

1. [§1.3.1 — Control problem on the Pontryagin–Hamilton equation](#131)
2. [§1.3.2 — Method for real-time ε-admissible trajectory synthesis (SVCM)](#132)
3. [Theorem 2 — realizational ε-admissibility and controllability](#thm2)
4. [§1.3.3 — Comparison with existing approaches](#133)
5. [§1.4 — Chapter 1 conclusion](#14)
6. [Ch. 2 opening / §2.1 — RL role, MDP, actor-critic (symbols used later)](#21)
7. [§2.1 cont. — MPC role in the actor](#21mpc)
8. [MPC → LQG projection (Kalman filter, Riccati, LQG law)](#lqg)
9. [§2.2 — MPC's role in constructing the behaviour plan (PMP correspondence)](#22mpc)
10. [Algorithm 1 — Синтез сценариев поведения агента](#alg1)
11. [§2.2 — Spatial-knowledge augmentation (start)](#22spatial)
12. [Symbol glossary](#glossary)

---

<a name="131"></a>
## 1. §1.3.1 — Постановка задачи управления на основе уравнения Понтрягина-Гамильтона

> p. 48

$$
\dot{x} = \frac{\partial H}{\partial \lambda}, \qquad \dot{\lambda} = -\frac{\partial H}{\partial x}
$$

The canonical (Hamiltonian) system for the pair $(x,\lambda)$, where $H$ is
the Pontryagin Hamiltonian and $\lambda(t)$ is the vector of Lagrange
multipliers on the dynamic constraints.

> p. 48

$$
u^{*}(t) \in \arg\min_{u} H(x^{*}(t), u, \lambda^{*}(t), t)
$$

Stationarity condition on the control: the optimal control minimizes (or
maximizes, depending on the sign convention) the Hamiltonian pointwise in
time. Text notes explicitly this may be a max instead of a min "depending on
the orientation of the problem."

The prose (p. 48) states that going from the variational (Lagrangian) form to
the Pontryagin–Hamilton form is, in essence, folding the dynamic constraints
into the functional and rewriting the stationarity conditions in canonical
Hamiltonian form for the pair $(x,\lambda)$, with $H$ playing the role of a
generalized Hamiltonian for the optimal-control problem.

### State and dynamics of the three-wheeled robot

> p. 48

$$
x(t) = \begin{pmatrix} x_c(t) \\ y_c(t) \\ \theta(t) \end{pmatrix}, \qquad
u(t) = \begin{pmatrix} v(t) \\ \omega(t) \end{pmatrix}
$$

State vector of a **three-wheeled robot with one driving wheel**: centre-of-mass
position $(x_c, y_c)$ and heading $\theta$; control is linear speed $v$ and
angular speed $\omega$. **This is the model the whole §1.3 development is
built on — a 3-state unicycle-like plant, distinct from this project's 5D
skid-steer wheel-space model.**

> p. 48

$$
\dot{x}_c(t) = v(t)\cos\theta(t), \qquad
\dot{y}_c(t) = v(t)\sin\theta(t), \qquad
\dot{\theta}(t) = \omega(t)
$$

Standard unicycle kinematics.

### The cost functional (1.7)

> p. 48–49

$$
J(v,\omega) = \frac{1}{2}\int_{t_0}^{t_f}
\Big( q_x(x_c(t)-x_{c,T})^2 + q_y(y_c(t)-y_{c,T})^2 + q_\theta(\theta(t)-\theta_T)^2
+ r_v v(t)^2 + r_\omega \omega(t)^2 \Big)\,dt
\tag{1.7}
$$

with $q_x, q_y, q_\theta, r_v, r_\omega > 0$. Terminal target state is
$x_T = (x_{c,T}, y_{c,T}, \theta_T)$, to be reached (approximately) at time
$t_f$ from an initial state $x(t_0) = x_0$.

**Gloss:** the running cost is a quadratic tracking penalty (position + heading
error) plus quadratic control-effort penalties on $v$ and $\omega$, weighted
by $r_v, r_\omega$. The prose explains the standard interpretation: larger
$r_v$ makes the controller more reluctant to increase speed; larger
$r_\omega$ makes it prefer smoother (lower-$\omega$) turns.

### Costate, Hamiltonian, and the explicit canonical system

> p. 49

$$
\lambda(t) = \begin{pmatrix} \lambda_1(t) \\ \lambda_2(t) \\ \lambda_3(t) \end{pmatrix}
$$

The costate (Lagrange-multiplier) vector, one component per state. The text
is explicit about notation choice: some Russian-language sources write
$\psi(t)$ for this; the dissertation deliberately uses the classical
$\lambda(t)$ and calls it "the multiplier that carries the dynamics'
coefficients."

> p. 49

$$
H(x,v,\omega,\lambda,t) = L(x,v,\omega,t) + \lambda_1 \dot{x}_c + \lambda_2 \dot{y}_c + \lambda_3 \dot{\theta}
$$

$$
L(x,v,\omega,t) = \frac{1}{2}\big(q_x(x_c-x_{c,T})^2 + q_y(y_c-y_{c,T})^2 + q_\theta(\theta-\theta_T)^2 + r_v v^2 + r_\omega \omega^2\big)
$$

$$
\dot{x}_c = v\cos\theta, \qquad \dot{y}_c = v\sin\theta, \qquad \dot{\theta}=\omega
$$

Pontryagin Hamiltonian = running Lagrangian $L$ + costate-weighted dynamics.
Substituting the dynamics into $H$ (p. 50):

> p. 50

$$
H = \frac{1}{2}\big(q_x(x_c-x_{c,T})^2 + q_y(y_c-y_{c,T})^2 + q_\theta(\theta-\theta_T)^2 + r_v v^2 + r_\omega \omega^2\big)
+ \lambda_1 v\cos\theta + \lambda_2 v\sin\theta + \lambda_3 \omega
$$

Fully expanded Hamiltonian used to derive the canonical equations below.

> p. 50 — canonical system on $[t_0,t_f]$

$$
\dot{x}^{*}(t) = \frac{\partial H}{\partial \lambda}(x^{*},v^{*},\omega^{*},\lambda^{*},t), \qquad
\dot{\lambda}^{*}(t) = -\frac{\partial H}{\partial x}(x^{*},v^{*},\omega^{*},\lambda^{*},t)
$$

with boundary condition $x^{*}(t_0)=x_0$ and terminal condition
$\lambda^{*}(t_f)=0$ (stated as the case for a purely integral functional,
i.e. no terminal cost term).

> p. 50 — explicit costate ODEs

$$
\dot{x}^{*}_c = v^{*}\cos\theta^{*}, \qquad
\dot{y}^{*}_c = v^{*}\sin\theta^{*}, \qquad
\dot{\theta}^{*} = \omega^{*}
$$

$$
\dot{\lambda}^{*}_1(t) = -q_x(x^{*}_c(t)-x_{c,T})
$$
$$
\dot{\lambda}^{*}_2(t) = -q_y(y^{*}_c(t)-y_{c,T})
$$
$$
\dot{\lambda}^{*}_3(t) = -q_\theta(\theta^{*}(t)-\theta_T) + \lambda^{*}_1(t)v^{*}(t)\sin\theta^{*}(t) - \lambda^{*}_2(t)v^{*}(t)\cos\theta^{*}(t)
$$

The costate dynamics: $\dot\lambda_1,\dot\lambda_2$ are simply the (negative,
weighted) position tracking errors; $\dot\lambda_3$ additionally couples back
through $v,\theta$ from the nonlinear kinematics.

> p. 50 — stationarity in control

$$
\frac{\partial H}{\partial v} = r_v v + \lambda_1\cos\theta + \lambda_2\sin\theta = 0, \qquad
\frac{\partial H}{\partial \omega} = r_\omega \omega + \lambda_3 = 0
$$

> p. 51 — optimal controls

$$
v^{*}(t) = -\frac{1}{r_v}\big(\lambda^{*}_1(t)\cos\theta^{*}(t) + \lambda_2(t)\sin\theta^{*}(t)\big), \qquad
\omega^{*}(t) = -\frac{1}{r_\omega}\lambda^{*}_3(t)
$$

for $t\in[t_0,t_f]$. **Gloss:** the optimal linear/angular speed is a linear
function of the costate, scaled inversely by the control-cost weight — the
textbook PMP closed-form for a quadratic-in-control Hamiltonian. Conclusion of
§1.3.1 (p. 51): solving for the optimal three-wheeled-robot control on
$[t_0,t_f]$ reduces to a **boundary-value problem for the ODE system** in
$x^{*}(t), \lambda^{*}(t)$, with $v^{*},\omega^{*}$ read off from $\lambda^{*},\theta^{*}$.

---

<a name="132"></a>
## 2. §1.3.2 — Метод синтеза $\varepsilon$-допустимой траектории в реальном времени

> p. 51

"Добавим в 1.6 внешние факторы реальной среды $\mu(t)$, например, эффективный
коэффициент трения и прочие свойства покрытия."

**Gloss:** the model gains an explicit environment-factor input $\mu(t)$ (e.g.
effective friction coefficient / surface properties). $u_{adm}$ denotes the
set of admissible controls, constrained by actuator power, thrust, angular
speed, and by the admissible region of the state's phase space.

> p. 51 — dynamics with environment factor

$$
\dot{x}(t) = f(x(t), u(t), \mu(t)), \qquad x(t_0)=x_0
$$

$$
J(u;\mu) = \Phi(x(t_f),\mu) + \int_{t_0}^{t_f} L(x(t),u(t),\mu)\,dt
$$

General running+terminal cost functional now explicitly parameterized by the
environment trajectory $\mu(\cdot)$. $\Phi$ is the terminal cost, $L$ the
running cost.

### Definition of $J^*$ and ε-optimality

> p. 51

$$
J^{*}(\mu) = \inf_{u\in u_{adm}} J(u;\mu)
$$

For a fixed environment trajectory $\mu(\cdot)$, $J^*(\mu)$ is the best
achievable value of the functional.

> p. 51–52

$$
J(\bar{u};\mu) \le J^{*}(\mu) + \varepsilon
$$

**Definition.** A control $\bar u(\cdot)$ is called **$\varepsilon$-optimal**
(or realizing an $\varepsilon$-max/min) on $u_{adm}$ if its cost differs from
the computed optimum by no more than $\varepsilon > 0$. Applied reading
(dissertation's own words, p. 52): for the three-wheeled platform this is
interpreted as a **bounded set of admissible agent states** within which the
robot can move **without critical effect on the final distance to the goal**.

**Gloss:** this is the formal ε-optimality that the CLAUDE.md "epsilon
optimality / SVCM" section refers to — $J[u] \le J^*[z] + \epsilon$ in the
notation used there. Confirmed: the set of ε-optimal controls is written
$U_\varepsilon$ in the prose narrative but the PDF itself writes the
inequality directly rather than naming the set symbolically on this page — no
`$U_\varepsilon$` symbol appears rendered in the equations on pp. 48–62; it is
only named in prose ("ε-допустимый набор управлений"). Treat "$U_\varepsilon$"
as the prose name for $\{\bar u : J(\bar u;\mu)\le J^*(\mu)+\varepsilon\}$,
not as dissertation notation.

### Decomposition into nominal + correction

> p. 52

$$
u_{adm} = u_J + \bar{u}
$$

The admissible control is split into $u_J$ — "the $J$-control," i.e. the
control the system already has from its existing functional/plan — plus a
correction $\bar u$ drawn from the $\varepsilon$-optimal control set when
needed. **Gloss:** this operationalizes the SVCM idea from CLAUDE.md — first
try to compensate with the control you already have ($u_J$), only escalate to
a server-supplied $\varepsilon$-optimal template ($\bar u$) if that fails.

### The SVCM method statement — offline catalogue

> p. 52

"Предположим, что на этапе офлайн-расчёта удалённый сервер построил конечное
семейство сценариев взаимодействия со средой для каждого фактора
$\{\mu^{(k)}(\cdot)\}_{k=1}^N$ — сухой асфальт, мокрое покрытие, лёд, грязь и т.д."

$$
u^{(k)}(\cdot) \in u_{adm}, \qquad
J(u^{(k)};\mu^{(k)}) \le J^{*}(\mu^{(k)}) + \varepsilon, \qquad
J^{*}(\mu^{(k)}) = \inf_{u\in u_{adm}} J(u;\mu^{(k)})
$$

For each of $N$ finite offline scenarios (dry asphalt, wet surface, ice, mud,
etc.), the server precomputes an $\varepsilon$-optimal control $u^{(k)}(\cdot)$
via PMP. These trajectories, together with their costate trajectories and
Hamiltonian parameters, are stored on the server as a **catalogue of
suboptimal crisis templates**, indexed by environment type and by deviation
magnitude from expected characteristics. **"Назовем такой подход
STRL-Variative control-method (SVCM)"** — this is the exact page where the
method is named.

### Online event/request loop

> p. 52–53

"Пусть агент движется в реальном времени и измеряет на каждом шаге $t_j$ свою
оценку состояния $\hat{x}(t_j)$ и набор диагностических признаков $\xi(t_j)$
(например, индикаторы проскальзывания колёс)."

On a traction-loss event, the agent sends a request at $t_j$ containing
$(\hat x(t_j), \xi(t_j))$.

> p. 53

$$
0 \le \tau_{\max} \ll t_f - t_0
$$

Communication delay is bounded by a known $\tau_{\max}$, assumed small
relative to the horizon length.

The server's reply, arriving at time $k_j$, has the form of an index
$\mu^{(k_j)}$ of the matching scenario and the instruction "use control
template $u^{(k_j)}$ on horizon $[t_j, t_j+T_{loc}]$", where $T_{loc}>0$ is a
fixed local control-window length (e.g. the time to drive out of a puddle).
Onboard storage holds either the parameterized trajectory $u^{(k_j)}(\cdot)$
itself, or a compact rule for reproducing it (cites ref [114]).

---

<a name="thm2"></a>
## 3. Теорема 2. О реализационной $\varepsilon$-допустимости и управляемости с учётом реального времени

> p. 53

**Названа "Теорема 2"** in the source. (CLAUDE.md's summary calls it "Theorem
1" — the source's own numbering is Theorem 2; this transcription preserves
the source numbering.)

### Условия применимости (three assumptions)

> p. 53–54

**1.** Environment scenarios sufficiently cover real situations: for every
realized environment trajectory $\mu(\cdot)$ and every problem-detection
instant $t_j$, there exists an index $k_j$ such that, for the functions $f,L$
and the current state $\hat x(t_j)$, the dynamics on horizon $[t_j,
t_j+T_{loc}]$ $\delta$-approximate the real dynamics with $\delta>0$ (error
does not exceed a pre-specified $\delta$).

**2.** Communication delay $\tau_j$ is small compared to $T_{loc}$ and the
system dynamics: using control $u^{(k_j)}(\cdot)$ shifted in time by $\tau_j$,
the deviation of the actual trajectory from the trajectory corresponding to
an ideal start at $t_j$ is also bounded above by a quantity of order
$\delta_\tau$, depending on $\tau_{\max}$ and the Lipschitz constant of $f$.

**3.** The admissible-control set $u_{adm}$ is consistent with the system's
physical description: all trajectories $u^{(k)}(\cdot)$ satisfy the physical
actuator and state constraints, and any local corrections needed for the time
shift and for binding to the current $\hat x(t_j)$ remain within $u_{adm}$.

**Gloss:** these are exactly the three hypotheses CLAUDE.md's SVCM section
paraphrases as "scenario coverage," "communication delay small relative to
$T_w$ with a Lipschitz drift bound," and "templates respect actuator/state
constraints."

### Conclusion — existence of $\varepsilon_{tot}$

> p. 54

"Тогда существует $\varepsilon_{tot}>0$, зависящее от исходного $\varepsilon$,
от погрешностей аппроксимации $\delta$, задержек $\delta_\tau$ и
продолжительности окна $T_{loc}$, такое, что для любой реализующейся
траектории среды $\mu(\cdot)$ и произвольной последовательности моментов
срабатывания детектора проскальзывания $t_j$ верно альтернативное утверждение:"

There exists $\varepsilon_{tot} > 0$ (depending on $\varepsilon,\delta,
\delta_\tau, T_{loc}$) such that, for any realized environment trajectory and
any sequence of slip-detector trigger times, the following **dichotomy**
holds:

### Branch 1 — Управляемость (controllable)

> p. 54

"Если для реальной задачи на интервале $[t_0,t_f]$ существует допустимое
управление $u^{*}(\cdot)\in u_{adm}$, обеспечивающее значение функционала
$J(u^{*};\mu)=J^{*}(\mu)$..."

$$
J(u_{real};\mu) \le J^{*}(\mu) + \varepsilon_{tot}
$$

If an admissible optimal control exists for the real problem on $[t_0,t_f]$,
then the real-time "event → server index selection → apply matching template
$u^{(k_j)}$ on local windows $[t_j,t_j+T_{loc}]$" strategy achieves
$\varepsilon_{tot}$-optimality — i.e. periodic server commutation with
precomputed $\varepsilon$-optimal recovery strategies guarantees the agent
recovers from local traction-loss events and, overall, performs no worse than
the optimal control, up to $\varepsilon_{tot}$.

### Branch 2 — Неуправляемость (uncontrollable)

> p. 55

"Если для траектории среды $\mu(\cdot)$ и заданных ограничений $u_{adm}$ не
существует никакого управления $u(\cdot)$, обеспечивающего приемлемое
значение функционала (например, $J(u;\mu)\le J_{\max}$ для заданного порога
качества), то никакая стратегия, основанная на выборке из конечного
$\{u^{(k)}\}$ и ограниченная реальным временем ответа сервера, также не может
обеспечить требуемый уровень качества."

**Gloss:** if no admissible control achieves an acceptable cost
($J(u;\mu)\le J_{\max}$ for a given quality threshold), no finite-catalogue,
real-time-bounded strategy can either. The system is a-priori unable to cope
with the environment change: **the cause is not computational distribution
and not communication delays, but the fundamental unreachability of the goal
under the given physical constraints.**

The concluding restatement (p. 55): either the problem is solvable in the
admissible-control class, in which case the agent-server distributed scheme
delivers $\varepsilon_{tot}$-close-to-optimal motion (recovery from puddles,
ice, other perturbation zones); or the required quality is unachievable by
any control in $u_{adm}$, in which case no precomputed $\varepsilon$-optimal
template, however selected/switched in real time, can restore the system's
equilibrium — a **fundamental** and not an **algorithmic** limitation. This is
illustrated by "Рисунок 2. Применение SVCM для трехколесной робототехнической
системы" (p. 56, a floor-plan sketch showing a start→end route with a
puddle/detection zone $\zeta(t_j)\to\mu^{(k_j)}$ triggering a template
$u^{(k_j)}$ applied over $[t_j, t_j+T_{loc}]$).

---

<a name="133"></a>
## 4. §1.3.3 — Основные отличия метода от существующих подходов

> pp. 56–57 — abstract existence theory (Урысона / Uryson)

$$
x(t) = x_0 + \int_{t_0}^{t} F(s,x(s),u(s))\,ds
$$

The classical strict-existence approach: system given by a Uryson integral
equation, admissible controls taken as a closed ball in $L_p$, and the payoff
functional lower semi-continuous.

> p. 57

$$
J(u_\varepsilon) \le \inf_u J(u) + \varepsilon
$$

For any $\varepsilon>0$ there exists $u_\varepsilon(\cdot)$ satisfying this —
the abstract existence-of-ε-optimal-control result the dissertation contrasts
itself against. **Gloss / contrast (pp. 57–58, prose only, no further
formulas):** this class of result is a pure existence proof from compactness +
lower-semicontinuity arguments — no agent-server architecture, no real-time
or network-delay accounting, no finite catalogue of precomputed strategies.
SVCM, by contrast, ties ε-optimality to a **concrete finite set of
trajectories** plus an event-triggered switching algorithm.

### Comparison with MDP ε-optimal policies

> p. 58 (prose, no numbered formula rendered as a standalone equation beyond the
inline symbols $\pi^*$, $O(\varepsilon)$)

"...переходные вероятности меняются из-за изменения величины награды в
пределах $\varepsilon$ и при достаточно малом $\varepsilon$ стратегия
$\pi^{*}$ остаётся $O(\varepsilon)$-оптимальной для возмущённой системы."

**Gloss:** a stationary optimal policy $\pi^*$, found for a reference/nominal
MDP, stays $O(\varepsilon)$-optimal under small perturbations to transition
probabilities. Distinguished from SVCM on the grounds that MDP results are for
**discrete states/actions**, average-cost criteria, and stationary policies —
no continuous dynamics, no real-time/delay accounting, no event-triggered
scenario switching. RL (as MDP ε-optimal-policy search) is explicitly framed
as **"the server-side part of the present approach"** (p. 58) — i.e. the paper
positions its own actor-critic Chapter 2 work as an instance of this MDP
literature, used to build the catalogue, not as a competitor to SVCM.

### Comparison with invariant-admissible-set switching methods

> pp. 59–60 (prose; the one rendered symbol is $X_{adm}^{(i)}$)

"...вычисляется множество состояний $X_{adm}^{(i)}$, из которых, следуя этому
режиму, система сможет в будущем удовлетворять всем ограничениям..."

For each control mode $i$ (e.g. a specific constrained linear controller), a
maximal admissible/invariant set $X_{adm}^{(i)}$ is precomputed; switching to
mode $i$ is permitted iff the current state belongs to $X_{adm}^{(i)}$.
**Gloss:** distinguished from SVCM in that these methods target **constraint
satisfaction at mode switches**, not ε-optimality of a cost functional; they
need an accurate environment model, and carry no catalogue of ε-optimal PMP
trajectories.

### Comparison with probabilistic/robust PMP

> p. 60

$$
\bar{H}(x,u,\lambda,t) = \mathbb{E}_\theta\big[L(x,u,t) + \lambda^{\bullet}f_\theta(x,u,t)\big]
$$

Mean/average Hamiltonian over an uncertainty distribution on a dynamics
parameter $\theta$ (e.g. an ensemble of neural ODEs or a Bayesian model),
giving rise to a "probabilistic PMP": minimizing $\bar H$ yields necessary
optimality conditions for the mean functional $\mathbb{E}[J]$, solved
numerically.

**Gloss (p. 61):** probabilistic PMP needs an uncertainty *distribution*
rather than an exact model, and averages robustness/uncertainty **inside one
continuous PMP problem**. SVCM instead does **not** average over uncertainty —
it "switches" between precomputed scenarios corresponding to discrete
environment-parameter values, using real-time telemetry to pick among them.
This is the paper's own closing contrast for the whole related-work section.

---

<a name="14"></a>
## 5. §1.4 — Вывод главы 1

> pp. 61–62 (prose conclusion, no new equations — synthesizes the chapter).
Key claims worth keeping traceable:

- LQ methods are noted as practical for lightweight/weak-compute systems via
  linearization around a working trajectory + quadratic cost on
  centre-of-mass coordinates, orientation, wheel angles/speeds and actuator
  torques → standard algebraic Riccati equations + stationary state-feedback
  law.
- Real-time full-model updates (Kalman-filter coefficients, Riccati re-solves)
  are argued impractical onboard under limited compute/comms → motivates the
  distributed architecture: heavy computation on a remote server, only a
  local regulator + mode-switch mechanism onboard.
- States that classical comm protocols (named explicitly: **DShot**) cannot
  carry full matrix/state-vector data in real time because of frame structure
  and frame-rate limits, hence the proposed **SFCC** — "a logical
  multiplexing layer over the existing frame format" for compact scenario
  packets (matrix bundle + scenario ID).
- States the PMP-based path (SVCM) as the chapter's conclusion: precompute a
  finite family of $\varepsilon$-optimal controls/trajectories for different
  environment scenarios via PMP, offline.

> p. 62 — closing line, restated for citation:

"В результате в главе сформулирована общая задача оптимального управления
мобильным агентом с учётом ограничений и неопределённости... предложена
архитектура распределённого управления, способная работать в реальном
времени при ограниченных каналах связи."

---

<a name="21"></a>
## 6. Глава 2 opening / §2.1 — MDP value functions, actor-critic (symbols used later)

> p. 64 — Bellman value functions

$$
V^{\pi}(s) = \mathbb{E}\Big[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t) \,\Big|\, s_0=s,\ a_t\sim\pi(\cdot|s_t)\Big]
$$

$$
V^{\pi}(s) = \mathbb{E}_{a\sim\pi(\cdot|s),\,s'\sim P(\cdot|s,a)}\big[r(s,a) + \gamma V^{\pi}(s')\big]
$$

$$
Q^{\pi}(s,a) = \mathbb{E}\Big[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\,\Big|\, s_0=s,\ a_0=a,\ a_{t>0}\sim\pi\Big]
$$

$$
Q^{\pi}(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'\sim P(\cdot|s,a)}\,Q^{\pi}(s',\pi(s'))
$$

Standard state-value $V^\pi$ and action-value $Q^\pi$ functions and their
Bellman equations. $s$ = environment state (platform position/velocity, scene
config); $s'$ = next state; $r$ = instantaneous reward; $\gamma\in[0,1)$ =
discount factor (closer to 1 ⇒ future reward matters more); $\pi$ = policy.

> p. 65 — policy-gradient estimator

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s\sim d^\pi,\, a\sim\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a|s)\, Q^{\pi}(s,a)\big]
$$

$\pi_\theta(a|s)$ = parametric policy; $\theta$ = its parameters (all
weights/biases); $J(\theta)$ = expected-return objective being maximized by
gradient ascent.

> p. 66 — actor update (advantage form)

$$
\nabla_\theta J(\theta) \approx \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a|s)\, A^{\pi}(s,a)\big], \qquad
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)
$$

$A^\pi$ = advantage estimate computed by the critic. Actor-critic is framed
as a variance-reduction device relative to plain policy gradient.

> p. 68 — discrete plant model for the three-wheeled agent (actor-critic framing)

$$
x_{k+1} = f(x_k, u_k)
$$

$x_k$ = state vector: centre-of-mass $(x,y)$, heading $\theta$, wheel
rotation angles/angular speeds, and (if needed) linear/angular platform
speed. $u_k$ = control vector, e.g. increments of linear/angular speed or
drive-wheel torques. Agent gets scalar reward $r_k$ each step reflecting
local goal quality (tracking, energy, collision avoidance).

> p. 69

$$
\pi_\theta(u|x), \qquad u_k = u_\theta(x_k)
$$

Deterministic-policy actor: parametric mapping from state to control.
$V_\phi(x_k)$ = critic's state-value estimate, $\phi$ = critic parameters.

---

<a name="21mpc"></a>
## 7. §2.1 cont. — Roll of MPC in the actor

> p. 69

$$
x_{k+1} = Ax_k + Bu_k
$$

Discrete model linearized around a working trajectory $(\bar x_k, \bar u_k)$
— here $x_k$ denotes **deviation from the reference trajectory**, not an
attractor point as in LQR (text explicitly flags this distinction).

> p. 70 — finite-horizon MPC cost

$$
J_k = x_{k+N}^{\bullet} P x_{k+N} + \sum_{i=0}^{N-1}\big(x_{k+i}^{\bullet} Q x_{k+i} + u_{k+i}^{\bullet} R u_{k+i}\big)
$$

subject to $x_{k+i+1}=Ax_{k+i}+Bu_{k+i},\ i=0,\dots,N-1$. ($x^\bullet$ denotes
transpose in this document's notation, i.e. $x^\top$.) $Q,R$ penalize state
deviation / control effort; terminal $P$ shapes asymptotic behaviour. Only the
first control $u_k=u_k^{*}(x_k)$ of the optimal sequence
$\{u_k^*,\dots,u_{k+N-1}^*\}$ is applied (receding horizon).

> p. 70 — MPC actor parameterized by $\theta^p$

$$
u_k = \Pi_{\theta^p}(x_k)
$$

$\Pi_{\theta^p}$ = the map realized by solving the MPC problem on horizon $N$
for the current state, with $Q,R,P,A,B$ depending on parameter vector
$\theta^p$.

> p. 71 — QP form of the linear MPC problem

$$
\min_{u_{0:N-1}} \Big[ x_N^{\bullet} P(\theta^p) x_N + \sum_{i=0}^{N-1}\big(x_i^{\bullet}Q(\theta^p)x_i + u_i^{\bullet}R(\theta^p)u_i\big)\Big]
$$

subject to $x_{i+1}=A(\theta^p)x_i+B(\theta^p)u_i,\ x_0=x_k$. Optimal sequence
$\{u_0^*,\dots,u_{N-1}^*\}$; actor applies $u_k=u_0^*(x_k;\theta^p)$.

> p. 71 — differentiable-MPC QP rewrite

$$
\min_z \tfrac{1}{2} z^{\bullet}H(\theta^p)z + f(x_k;\theta^p)^{\bullet}z
$$

subject to $G(\theta^p)z \le h(\theta^p)$. $z$ packs all controls+states over
the horizon; $H(\theta^p)$, $f(x_k;\theta^p)$ depend on dynamics and cost
matrices; $G$ encodes how state/control enter constraints, $h$ the numeric
bounds. KKT conditions give a system for the optimal $z^*$ and multipliers
$\lambda^*$.

> p. 71–72 — implicit-function / KKT differentiation

$$
\Phi(z^{*},\lambda^{*};x_k,\theta^p) = 0
$$

$$
\frac{\partial z^{*}}{\partial \theta^p} = -\Big(\frac{\partial \Phi}{\partial(z,\lambda)}\Big)^{-1}\frac{\partial \Phi}{\partial \theta^p}
$$

Implicit-function-theorem differentiation of the fixed active-constraint-set
KKT system, from which $\partial u_k/\partial\theta^p$ is extracted (since
$u_k$ is the first component of $z^*$).

> p. 72 — gradient of the outer objective w.r.t. actor parameters

$$
\nabla_{\theta^p} J = \sum_k \frac{\partial J}{\partial u_k}\frac{\partial u_k}{\partial \theta^p} = \sum_k \frac{\partial J}{\partial u_k}\frac{\partial \Pi_{\theta^p}(x_k)}{\partial \theta^p}
$$

Chain rule through the differentiable MPC solver, given the critic's gradient
estimate $\partial J/\partial u_k$. This is what makes $Q(\theta^p)$,
$R(\theta^p)$, $P(\theta^p)$ (and the linear model $A,B$, plus any
neural-net dynamics extension) trainable end-to-end.

---

<a name="lqg"></a>
## 8. MPC → LQG projection (Kalman filter, Riccati, LQG law)

> p. 73 — standard linear plant + infinite-horizon MPC cost (repeated form used
for the LQR derivation)

$$
x_{k+1} = Ax_k + Bu_k
$$

$$
J_k = x_{k+N}^{\bullet}Px_{k+N} + \sum_{i=0}^{N-1}\big(x_{k+i}^{\bullet}Qx_{k+i}+u_{k+i}^{\bullet}Ru_{k+i}\big)
$$

> p. 73 — discrete algebraic Riccati equation (control)

$$
P_\infty = A^{\bullet}P_\infty A - A^{\bullet}P_\infty B(R+B^{\bullet}P_\infty B)^{-1}B^{\bullet}P_\infty A + Q
$$

Initialized/anchored using the MPC's own terminal matrix $P$.

> p. 73 — stationary gain and resulting infinite-horizon cost

$$
K = (R+B^{\bullet}P_\infty B)^{-1}B^{\bullet}P_\infty A
$$

$$
J_{LQR} = \sum_{k=0}^{\infty}\big(x_k^{\bullet}Qx_k + u_k^{\bullet}Ru_k\big)
$$

This law minimizes $J_{LQR}$ for the **same** $Q,R$ trained through the
differentiable MPC actor. Prose (p. 73): with no constraints, MPC and LQR are
cost-equivalent; MPC solves on a sliding finite horizon, LQR on an infinite
one — so transplanting the MPC-trained $Q,R$ (and partially $P$) into LQR
gives a linear regulator whose average strategy matches what the MPC actor
learned within its horizon.

> p. 74 — estimation Riccati equation (Kalman filter)

$$
S = ASA^{\bullet} - ASC^{\bullet}(CSC^{\bullet}+V)^{-1}CSA^{\bullet} + W
$$

$W$ = process-noise covariance, $V$ = measurement-noise covariance, both
estimated during MPC training; $C$ = output/measurement matrix (implicit from
context — **not explicitly defined on this page**; standard LQG notation
$y=Cx+v$ is assumed but not stated).

> p. 74 — stationary Kalman gain

$$
L = SC^{\bullet}(CSC^{\bullet}+V)^{-1}
$$

> p. 74 — full LQG regulator

$$
\hat{x}_{k+1} = A\hat{x}_k + Bu_k + L\big(y_{k+1} - C(A\hat{x}_k+Bu_k)\big), \qquad
u_k = -K\hat{x}_k
$$

**Gloss:** the complete separation-principle LQG controller — a Kalman-filter
state estimator feeding a certainty-equivalent LQR gain $K$. This closes
§2.1's arc: train a differentiable-MPC actor with a critic, then "project"
its converged $Q,R$ (and the identified noise covariances $W,V$) onto a
classical, cheap-to-run LQG regulator — sacrificing the receding-horizon
optimization at each step for a fixed linear law, suitable for a low-power
onboard computer.

---

<a name="22mpc"></a>
## 9. §2.2 — Роль МРС в построении плана поведения в вариационной задаче

> p. 75 — server-side linearized discrete model per scenario $s$

$$
x_{k+1} = A^{(s)}x_k + B^{(s)}u_k + d_k^{(s)}
$$

Obtained by linearizing the continuous model around a nominal trajectory for
a fixed scenario $s$, reconstructed either from PMP data or directly from the
actor-critic architecture. The scenario index $s$ ranges over **nominal**
scenarios $\mu(u_J)$ (motion to goal under different surface types) and
**crisis** scenarios $\mu(\bar u)$ (recovery from unforeseen environment
changes: puddle, ice, noisy comms).

> p. 76 — MPC quadratic criterion per scenario $s$

$$
J_{MPC}^{(s)} = x_N^{\bullet}Q_f^{(s)}x_N + \sum_{k=0}^{N-1}\big(x_k^{\bullet}Q^{(s)}x_k + u_k^{\bullet}R^{(s)}u_k\big), \qquad
Q^{(s)}\succeq 0,\ Q_f^{(s)}\succeq 0,\ R^{(s)}\succ 0
$$

These matrices are the discrete-time realization of the trained parameter
vector $\theta^p$; $Q^{(s)}$ sets the priority on position error, orientation
error, control-rate penalty, constraint margin, etc. — fixed after training.

> p. 76 — continuous functional for the PMP horizon

$$
J^{(s)}(x(\cdot),u(\cdot);\theta^p) = \int_{t_0}^{t_f}\big(x(t)^{\bullet}Q_c^{(s)}x(t)+u(t)^{\bullet}R_c^{(s)}u(t)\big)dt + x(t_f)^{\bullet}P_f^{(s)}x(t_f)
$$

Matrices $Q_c^{(s)}, R_c^{(s)}, P_f^{(s)}$ are chosen so the discrete
per-step MPC criterion is consistent with this integral form, e.g. via a
linear approximation (p. 76):

$$
Q_c^{(s)} \approx \frac{Q^{(s)}}{\Delta t}, \qquad R_c^{(s)} \approx \frac{R^{(s)}}{\Delta t}, \qquad P_f^{(s)} \approx Q_f^{(s)}
$$

$\Delta t$ = the RL/MPC discretization step. $Q_c,R_c$ set how strongly the
system penalizes state/control deviation **per unit time**; $P_f$ sets the
stiffness of the terminal penalty for missing the target state at $t_f$.

> p. 77 — PMP Hamiltonian and terminal condition per scenario

$$
H^{(s)}(x,u,\lambda,t;\theta^p) = x^{\bullet}Q_c^{(s)}x + u^{\bullet}R_c^{(s)}u + \lambda^{\bullet}f^{(s)}(x,u,\zeta^{(s)}(t))
$$

$$
\lambda^{(s)}(t_f) = 2P_f^{(s)}x(t_f)
$$

$\zeta^{(s)}(t)$ describes the fixed external-environment profile for scenario
$s$ (e.g. a reduced-$\mu_{eff}$ stretch between times $t_a,t_b$ — **not
formally defined elsewhere on these pages beyond this parenthetical**). The
terminal costate condition is stated for the quadratic-terminal-cost case.

Server storage for **nominal** scenarios $j\in J$ indexing $\mu(u_j)$: PMP
solution trajectories $u^{(j)}(t)$ and costate trajectories $\lambda^{(j)}(t)$
(p. 77), stored either as parametric trajectories or as compact approximators
(e.g. small networks realizing $u^{(j)}(t,x)$). On the agent side these are
the **nominal strategies** realizing ε-optimal tracking under the planned
dynamics/traction model.

> p. 78 — crisis-scenario counterparts

For each crisis scenario $\bar k\in\bar K$ indexing $\mu(\bar u)$, the server
sets its own matrix set $\bar Q_c^{(\bar k)}, \bar Q_f^{(\bar k)},
\bar R_c^{(\bar k)}$ in the discrete MPC, under a stricter criterion:
larger yaw-angle $\theta$ penalties, lateral drift-acceleration penalty,
large wheel-torque penalty, and possibly a penalty on deviating from a safety
corridor around the trajectory. After passing to the continuous form on
$[t_0,t_f]$: matrices $\bar Q_c^{(\bar k)}, \bar R_c^{(\bar k)}, \bar
P_f^{(\bar k)}$, defining the Hamiltonian $\bar H^{(\bar k)}$ and terminal
condition $\bar\lambda^{(\bar k)}(t_f)$. The PMP solutions
$\bar u^{(\bar k)}(t)$, $\bar\lambda^{(\bar k)}(t)$ form the **catalogue of
crisis templates**: puddle exit, post-ice-entry stabilization, recovery of
stability after hard braking on a slippery surface, etc.

> p. 78 — state-vector structure with actuator currents

$$
x = [x, y, \theta, v, \omega, i_1, i_2]^{\bullet}
$$

$i_1, i_2$ = drive-wheel actuator currents. For **nominal** scenarios
$Q_c^{(s)}$ emphasizes small $x,y,\theta$ error relative to the reference
trajectory, moderate speed penalty, weak current penalty. For **crisis**
scenarios $\hat Q_c^{(\bar k)}$ carries much larger weight on $\omega$ and on
wheel-speed differences (to suppress skid/uncontrolled slip), while $\hat
R_c^{(\bar k)}$ increases the control-effort penalty to limit aggressive
action in low-traction/noisy-signal zones.

**Gloss (link to §1.3.2):** the vector $\theta^p$ fixed by training uniquely
determines the discrete MPC cost matrices, which via the consistency
procedure above determine the continuous PMP matrices — so nominal scenarios
$\mu(u_j)$ and crisis scenarios $\mu(\bar u)$ are, server-side, **different
instances of the same PMP problem** for the three-wheeled platform, differing
in $Q_c,R_c,P_f$ and possibly in $A_c,B_c$ (capturing external-factor effects
on the local linear model). Onboard: use the current nominal scenario
$\mu(u_j)$ normally; on slip/traction-loss detection, request a crisis index
$\bar k\in\mu(\bar u)$ from the server, obtain the matching PMP solution (or
its compact parameterization), and execute it on the local time window while
staying inside $u_{adm}$ and $x_{adm}$.

---

<a name="alg1"></a>
## 10. Алгоритм 1. Синтез сценариев поведения агента

> pp. 78–83

### Этап 1. Синтез управления с помощью актор-критик метода

**1.** Continuous three-wheeled-platform model:
$$
\dot{x}(t) = f(x(t), u(t))
$$

**2.** Time discretization for the RL training environment, step $\Delta t$:
$$
x_{k+1} = f_d(x_k, u_k), \qquad x_k \approx x(t_k)
$$

**3.** Reward function reflecting policy quality:
$$
r_k = r(x_k, u_k)
$$
e.g. quadratic pose-error + energy cost:
$$
r_k = -\big((x_k - x_k^{ref})^{\bullet}Q_{act\text{-}cr}(x_k - x_k^{ref}) + u_k^{\bullet}R_{act\text{-}cr}u_k\big)
$$
$x_k^{ref}$ = desired trajectory (position, orientation) for the platform.
$Q_{act\text{-}cr}, R_{act\text{-}cr}$ = weight matrices for the RL reward
(**distinct from $Q^{(s)},R^{(s)}$** used later — subscripted "act-cr" for
"actor-critic").

**4.** Derive a policy $\pi_{\theta_{ppo}}(u|x)$ via an actor-critic algorithm
(subscript "$ppo$" appears rendered here, suggesting PPO as the concrete
algorithm used at this stage — **not otherwise stated explicitly in the
surrounding prose on this page**), obtaining for each state $x$ an
approximately-optimal control:
$$
u_{act\text{-}cr}(x) \approx \pi_{\theta_{act\text{-}cr}}(x)
$$
giving good simulated behaviour across a set of environment scenarios.

**5.** From the trained results, build a quadratic-functional approximation
consistent with the agent's policy, e.g. a local model:
$$
J_{loc}(x,u) \approx (x-x^{ref})^{\bullet}Q_0(x-x^{ref}) + (u-u_{act\text{-}cr}(x))^{\bullet}R_0(u-u_{act\text{-}cr}(x))
$$
$Q_0, R_0$ fitted by regression on RL-agent trajectory statistics (error and
control-effort data). These give the **initial** quadratic cost structure
for the subsequent MPC actor.

### Этап 2. Адаптация под реальную среду с MPC на акторе

**6.** Use the discrete platform model for MPC:
$$
x_{k+1} = f_d(x_k, u_k), \qquad u_k \in u_{adm}
$$
where $u_{adm}, x_{adm}$ reflect real thrust/angular-speed/phase-space
constraints.

**7.** In the actor, set an MPC problem on horizon length $N$ with a
quadratic discrete criterion:
$$
J_{MPC}(\theta^p) = x_N^{\bullet}Q_f(\theta^p)x_N + \sum_{k=0}^{N-1}\big(x_k^{\bullet}Q(\theta^p)x_k + u_k^{\bullet}R(\theta^p)u_k\big)
$$
where $Q(\theta^p), Q_f(\theta^p), R(\theta^p)$ are MPC matrices
parameterized by $\theta^p$; initialized as $Q(\theta^{p_0})\approx Q_0$,
$R(\theta^{p_0})\approx R_0$ (carrying over Step 5's fit).

**8.** In the actor-critic scheme with the MPC actor, interact with a more
realistic model (or the real platform):
- the critic estimates the long-term return $V(x)$ from real or simulated data;
- the actor, each step, solves the MPC problem
  $$
  x_{k+1} = f_d(x_k,u_k), \qquad u_k\in u_{adm}
  $$
  minimizing $J_{MPC}(\theta^p)$, and applies the resulting $u_k$;
- via the gradient w.r.t. $\theta^p$ (through the differentiable MPC), the
  weights in $Q(\theta^p), Q_f(\theta^p), R(\theta^p)$ are corrected,
  adapting to the real dynamics and environmental disturbances.

**9.** On completion of adaptation training, fix the MPC matrices for
different scenarios:
- for **nominal** scenarios $s\in S_j$ (set $\mu(u_j)$):
  $$
  Q^{(s)},\ Q_f^{(s)},\ R^{(s)}
  $$
- for **crisis** scenarios $\bar s\in S_{cr}$ (set $\mu(\bar u)$):
  $$
  \bar{Q}^{(\bar s)},\ \bar{Q}_f^{(\bar s)},\ \bar{R}^{(\bar s)}
  $$

These matrices now reflect both the RL agent's preferences and its
adaptation to the real environment via MPC.

### Этап 3. Переход к интегральной форме для MPC и передача матриц в ПМП

**10.** For each scenario $s$, set a time interval $[t_0^{(s)}, t_f^{(s)}]$
and a step count $N^{(s)}$ such that:
$$
t_f^{(s)} - t_0^{(s)} = N^{(s)} \cdot \Delta t
$$

**11.** From the discrete MPC matrices for scenario $s$ — $Q^{(s)}, Q_f^{(s)},
R^{(s)}$ — reconstruct the continuous cost matrices for the integral functional:
$$
Q_c^{(s)} = \frac{1}{\Delta t}Q^{(s)}, \qquad R_c^{(s)} = \frac{1}{\Delta t}R^{(s)}, \qquad P_f^{(s)} = Q_f^{(s)}
$$
and analogously for crisis scenarios $\bar s$:
$$
\bar{Q}_c^{(\bar s)} = \frac{1}{\Delta t}\bar{Q}^{(\bar s)}, \qquad \bar{R}_c^{(\bar s)} = \frac{1}{\Delta t}\bar{R}^{(\bar s)}, \qquad \bar{P}_f^{(\bar s)} = \bar{Q}_f^{(\bar s)}
$$

**12.** Form the integral PMP criterion for each scenario $s$ on
$[t_0^{(s)}, t_f^{(s)}]$:
$$
J^{(s)} = \int_{t_0^{(s)}}^{t_f^{(s)}} \big(x(t)^{\bullet}Q_c^{(s)}x(t) + u(t)^{\bullet}R_c^{(s)}u(t)\big)\,dt + x(t_f^{(s)})^{\bullet}P_f^{(s)}x(t_f^{(s)})
$$
and for crisis scenarios:
$$
\bar{J}^{(\bar s)} = \int_{t_0^{(\bar s)}}^{t_f^{(\bar s)}} \big(x(t)^{\bullet}\bar{Q}_c^{(\bar s)}x(t) + u(t)^{\bullet}\bar{R}_c^{(\bar s)}u(t)\big)\,dt + x(t_f^{(\bar s)})^{\bullet}\bar{P}_f^{(\bar s)}x(t_f^{(\bar s)})
$$

**13.** These matrices $Q_c^{(s)}, R_c^{(s)}, P_f^{(s)}$ (and
$\bar Q_c^{(\bar s)}, \bar R_c^{(\bar s)}, \bar R^{(\bar s)}$) are substituted
into Pontryagin's-maximum-principle Hamiltonian for the three-wheeled
platform:

Hamiltonian for scenario $s$:
$$
H^{(s)}(x,u,\lambda,t) = x^{\bullet}Q_c^{(s)}x + u^{\bullet}R_c^{(s)}u + \lambda^{\bullet}f^{(s)}(x,u,\zeta^{(s)}(t))
$$

terminal condition:
$$
\lambda(t_f^{(s)}) = \frac{\partial}{\partial x}\big(x^\top P_f^{(s)}x\big)\Big|_{x=x(t_f^{(s)})} = 2P_f^{(s)}x(t_f^{(s)})
$$

and analogously for crisis scenarios with $\bar Q_c^{(\bar s)}, \bar
R_c^{(\bar s)}, \bar P_f^{(\bar s)}$.

**Summary (p. 84, prose):** this is explicitly called a "three-stage
algorithm" — the RL algorithm sets a "behavioural" reference, the MPC actor
adapts the quadratic matrices $Q,Q_f,R$ to the real dynamics of the
three-wheeled platform, and then, via transition to the integral form, these
matrices become part of the PMP functional used to compute the catalogued
scenario PMP trajectories.

---

<a name="22spatial"></a>
## 11. §2.2 — Метод пополнения пространственных знаний (start)

> p. 84

Discusses joint refinement of the agent's "world model" and control policy —
the synthesis mechanism's structure stays fixed, only the parameters
describing environment dynamics/physical properties get refined online, when
the platform crosses terrain with a previously-unseen response class.

$$
\dot{x} = f(x,u,\zeta)
$$

Base dynamics equation, restated with $\zeta$ = the platform's own
diagnostic features (as in §1.3.2) from analyzing actuator behaviour.
For each recognized environment-response class $\varphi_i$, a parameterization
$\mu(\varphi_i)$ (a movement scenario for that recognized class) is assumed
known.

**On crossing a region with unknown response factors $\varphi_{new}$**: the
observed dynamics systematically diverges from the model's predictions
(inertia off, braking distance mismatched, more wheel slip, comms
noise/loss). This triggers, in the continuous platform-server architecture, a
mechanism for **identifying new environment parameters and extending the
spatial ontology** linking centre-of-mass coordinates to environment types
and their models.

> p. 84 — slow environment-parameter dynamics

$$
\dot{\varphi}_{env}(t) = g(\varphi_{env}(t), \zeta(t))
$$

$\varphi_{env}(t)$ = current estimate of the surface's physical
characteristics, evolving by a "slow" law $g$. **($g$ itself is not further
specified on this page.)**

The identifier updates $\varphi_{env}$ as data arrives, compensating for the
model/reality mismatch while the agent keeps executing its optimal control
under the previously-assumed model. If a region's dynamics fits no known
class $\varphi_i$ well, the identifier both corrects numerical values of
$\mu$ and other parameters, and **initiates creation of a new area type**
$\varphi_{new}$ in the spatial knowledge base — a new layer/label class tied
to the coordinate region where atypical behaviour was observed.

> p. 84 → carries onto p. 85 (out of the requested range; noted here because it
begins on p. 84): a prediction-error threshold criterion

$$
\varepsilon(t) = \|\tilde{x}(t+\Delta t) - x(t+\Delta t)\|
$$

$\hat x(t+\Delta t)$ (labelled in prose, though the equation itself uses
$\tilde x$) = model-predicted state; $x(t+\Delta t)$ = actually measured
state (sensor system). Sustained excess over a threshold $\varepsilon(t) >
\varepsilon_{crit}$ over a long-enough window, while the current map region
was previously of a known type $\varphi_i$, triggers the identifier to treat
this as detection of a new environment structure/regime. This launches a
local parameter-estimation problem for $\varphi_{env}$ (minimizing a
model-vs-observation mismatch functional), and — given enough data and a
persistent divergence from known classes — creates a new parametric class
$\varphi_{new}$ with reference model $\dot x_{new} = f(x,u,\zeta_{env})$ and a
spatial-region binding $\Omega_{new}\subset\mathbb{R}^2$.

**This is beyond the requested 74–84 detail range on its tail end** (the
$\varepsilon(t)$ threshold formula and the $\Omega_{new}$ binding render on
what is the start of p. 85 content flowing from a p. 84 paragraph break in the
PDF's own pagination); included here because it completes the sentence begun
on p. 84 and defines symbols ($\varphi_{env}$, $\Omega_{new}$) introduced on
p. 84 itself.

---

<a name="glossary"></a>
## 12. Symbol glossary

| Symbol | Meaning | First seen |
| --- | --- | --- |
| $x_c, y_c, \theta$ | centre-of-mass position and heading (3-wheel unicycle model) | p. 48 |
| $v, \omega$ | linear / angular speed (control) | p. 48 |
| $q_x,q_y,q_\theta,r_v,r_\omega$ | quadratic-cost weights, §1.3.1 | p. 48 |
| $\lambda(t)=(\lambda_1,\lambda_2,\lambda_3)$ | costate / Lagrange-multiplier vector | p. 49 |
| $H$ | Pontryagin Hamiltonian | p. 48 |
| $t_0,t_f$ | interval endpoints | p. 48 |
| $x_T=(x_{c,T},y_{c,T},\theta_T)$ | terminal target state | p. 48 |
| $\mu(t)$ | external environment factor (e.g. effective friction) | p. 51 |
| $u_{adm}$ | admissible control set | p. 51 |
| $x_{adm}$ | admissible state set | p. 75 |
| $J(u;\mu)$, $J^*(\mu)$ | cost functional and its infimum over $u_{adm}$ | p. 51 |
| $\varepsilon$ | ε-optimality tolerance | p. 51 |
| $u_J$ | the control already available from the existing functional | p. 52 |
| $\bar u$ | correction control drawn from the ε-optimal set | p. 52 |
| $\mu^{(k)}(\cdot)$, $N$ | catalogue of $N$ offline environment scenarios | p. 52 |
| $\hat x(t_j)$ | agent's state estimate at sample time $t_j$ | p. 52 |
| $\xi(t_j)$ | diagnostic features (e.g. wheel-slip indicators) | p. 52 |
| $\tau_{\max}, \tau_j$ | comms-delay bound / realized delay | p. 53 |
| $k_j$ | server-selected scenario index at request $j$ | p. 53 |
| $T_{loc}$ | fixed local control-window length | p. 53 |
| $\delta, \delta_\tau$ | dynamics-approximation error / delay-induced drift bound (Theorem 2 assumptions) | p. 53–54 |
| $\varepsilon_{tot}$ | total realized-time ε-optimality bound (Theorem 2 conclusion) | p. 54 |
| $J_{\max}$ | acceptable-quality cost threshold (uncontrollable branch) | p. 55 |
| $\theta$ (as uncertainty param) | random/unknown dynamics parameter (probabilistic PMP, §1.3.3) | p. 60 — **do not confuse with heading $\theta$ or policy params $\theta$/$\theta^p$** |
| $s,a,r,\gamma,\pi$ | MDP state, action, reward, discount, policy | p. 64 |
| $V^\pi, Q^\pi, A^\pi$ | state-value, action-value, advantage | p. 64–66 |
| $\theta$ (policy params) | actor's trainable parameters (§2.1) | p. 65 — **overloaded with heading and the uncertainty param above; disambiguate by context** |
| $\phi$ | critic's trainable parameters | p. 69 |
| $A,B$ | linearized discrete plant matrices | p. 69 |
| $Q,R,P$ | MPC stage / control / terminal cost matrices | p. 70 |
| $N$ | MPC horizon length (steps) | p. 70 |
| $\theta^p$ | vector of trainable MPC-matrix parameters | p. 70 |
| $\Pi_{\theta^p}$ | MPC-solve operator (the differentiable actor) | p. 70 |
| $z, H(\theta^p), f(x_k;\theta^p)$ | QP-form decision vector, Hessian, linear term | p. 71 |
| $G,h$ | linear-constraint matrix / bound vector | p. 71 |
| $\lambda^*$ (QP) | KKT multipliers of the MPC QP — **not the PMP costate**, overloaded symbol | p. 71 |
| $W, V$ | process- / measurement-noise covariances (Kalman filter) | p. 74 |
| $C$ | output/measurement matrix — used but not explicitly defined on p. 74 (standard $y=Cx+v$ assumed) | p. 74 |
| $S, L$ | estimation Riccati solution / stationary Kalman gain | p. 74 |
| $K$ | stationary LQR gain | p. 73 |
| $A^{(s)}, B^{(s)}, d_k^{(s)}$ | per-scenario linearized model (with offset/disturbance term $d_k^{(s)}$) | p. 75 |
| $s$ (scenario index), $\mu(u_j)$, $\mu(\bar u)$ | scenario index; nominal-scenario set; crisis-scenario set | p. 75 |
| $Q^{(s)},Q_f^{(s)},R^{(s)}$ | per-scenario discrete MPC cost matrices | p. 76 |
| $Q_c^{(s)},R_c^{(s)},P_f^{(s)}$ | per-scenario continuous PMP cost matrices | p. 76 |
| $\Delta t$ | RL/MPC discretization step | p. 76 |
| $\zeta^{(s)}(t)$ | fixed external-environment profile for scenario $s$ | p. 77 — **not formally defined beyond a parenthetical example** |
| $j\in J$ | index set of nominal scenarios | p. 77 |
| $u^{(j)}(t), \lambda^{(j)}(t)$ | stored nominal PMP trajectory / costate | p. 77 |
| $\bar k\in\bar K$ | index set of crisis scenarios | p. 78 |
| $\bar Q_c^{(\bar k)},\bar R_c^{(\bar k)},\bar P_f^{(\bar k)}$ | crisis-scenario continuous cost matrices | p. 78 |
| $i_1,i_2$ | drive-wheel actuator currents (extended state vector) | p. 78 |
| $Q_{act\text{-}cr}, R_{act\text{-}cr}$ | RL-reward weight matrices (Algorithm 1, Step 3) — **distinct from $Q^{(s)},R^{(s)}$** | p. 79 |
| $\pi_{\theta_{ppo}}$ / $u_{act\text{-}cr}$ | actor-critic policy (subscript suggests PPO) / its induced control | p. 80 |
| $Q_0, R_0$ | regression-fitted initial local quadratic-cost matrices (Algorithm 1, Step 5) | p. 80 |
| $\varphi_i, \varphi_{new}$ | recognized / newly-identified environment-response class | p. 84 |
| $\varphi_{env}(t)$ | online estimate of surface physical characteristics | p. 84 |
| $g(\varphi_{env},\zeta)$ | "slow" evolution law for $\varphi_{env}$ — **not further specified on p. 84** | p. 84 |
| $\varepsilon_{crit}$ | prediction-error alarm threshold (spatial-knowledge augmentation) — **overloaded with the ε-optimality tolerance above; unrelated quantity** | p. 84–85 |
| $\Omega_{new}\subset\mathbb{R}^2$ | spatial region bound to a newly-created environment class | p. 84–85 |

**Notation conventions observed across both ranges:** the source uses a raised
dot $x^{\bullet}$ for matrix/vector transpose throughout Chapter 2 (rendered
here as $x^{\bullet}$ to match the page exactly; interpret as $x^\top$).
$\varepsilon$ is **overloaded twice** — once as the ε-optimality tolerance
(§1.3.2–1.3.3) and once, unrelated, as the prediction-error threshold quantity
in §2.2 (p. 84–85, written $\varepsilon(t)$ and $\varepsilon_{crit}$). $\theta$
is **overloaded three ways** — heading angle (§1.3.1), an uncertainty
parameter in the probabilistic-PMP aside (§1.3.3), and actor/policy
parameters (§2.1 onward, later specialized to $\theta^p$ for MPC-matrix
parameters). No symbol in the transcribed ranges was found illegible at the
rendered resolution.
