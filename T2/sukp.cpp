// =============================================================================
//  Tarea 2 - CIT3352  Algoritmos Exactos y Metaheuristicas
//  Set-Union Knapsack Problem (SUKP)
//
//  Implementa:
//    (1) Greedy determinista
//    (2) Greedy estocastico (GRASP, con semilla -> reproducible)
//    (3) Algoritmo de trayectoria: Simulated Annealing (SA)
//    (4) Algoritmo de poblacion: Algoritmo Genetico (GA, memetico con reparacion)
//
//  Exporta resultados a CSV para su posterior analisis grafico en Python.
//
//  Grupo de 3 integrantes.
// =============================================================================
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

// -----------------------------------------------------------------------------
//  Instancia del problema
// -----------------------------------------------------------------------------
struct Instance {
    int m = 0, n = 0;          // # alternativas, # recursos
    ll  B = 0;                 // capacidad maxima
    vector<ll> p;              // beneficio de cada alternativa  p_i
    vector<ll> w;              // peso/costo de cada recurso     w_j
    vector<vector<int>> itemRes;   // recursos requeridos por cada alternativa
    vector<ll> selfCost;       // costo de los recursos de la alternativa por si sola
    string name;
};

Instance readInstance(const string& path, const string& name) {
    ifstream in(path);
    if (!in) { cerr << "No se pudo abrir " << path << "\n"; exit(1); }
    Instance I; I.name = name;
    ll ne;
    in >> I.m >> I.n >> ne >> I.B;
    I.p.resize(I.m);
    for (auto& x : I.p) in >> x;
    I.w.resize(I.n);
    for (auto& x : I.w) in >> x;
    I.itemRes.assign(I.m, {});
    for (ll k = 0; k < ne; ++k) {
        int i, j; in >> i >> j;
        I.itemRes[i].push_back(j);
    }
    I.selfCost.assign(I.m, 0);
    for (int i = 0; i < I.m; ++i)
        for (int j : I.itemRes[i]) I.selfCost[i] += I.w[j];
    return I;
}

// -----------------------------------------------------------------------------
//  Estado de una solucion con actualizacion incremental de costo
//    rcount[j] = # de alternativas seleccionadas que usan el recurso j
//    cost      = suma de pesos de los recursos en la UNION (cada uno una vez)
// -----------------------------------------------------------------------------
struct State {
    const Instance* I = nullptr;
    vector<char> sel;          // sel[i] = 1 si la alternativa i esta seleccionada
    vector<int>  rcount;       // # selecc. que usan el recurso j
    ll cost = 0, benefit = 0;

    void init(const Instance* inst) {
        I = inst;
        sel.assign(I->m, 0);
        rcount.assign(I->n, 0);
        cost = 0; benefit = 0;
    }
    // costo marginal de agregar i (peso de recursos aun no pagados)
    ll marginalAdd(int i) const {
        ll mc = 0;
        for (int j : I->itemRes[i]) if (rcount[j] == 0) mc += I->w[j];
        return mc;
    }
    // costo liberado al quitar i (recursos que solo usa i)
    ll freedRemove(int i) const {
        ll fc = 0;
        for (int j : I->itemRes[i]) if (rcount[j] == 1) fc += I->w[j];
        return fc;
    }
    void add(int i) {
        for (int j : I->itemRes[i]) { if (rcount[j] == 0) cost += I->w[j]; rcount[j]++; }
        sel[i] = 1; benefit += I->p[i];
    }
    void remove(int i) {
        for (int j : I->itemRes[i]) { rcount[j]--; if (rcount[j] == 0) cost -= I->w[j]; }
        sel[i] = 0; benefit -= I->p[i];
    }
    bool feasible() const { return cost <= I->B; }

    // Reconstruye costo/beneficio desde un vector sel dado (para decodificar cromosomas)
    void setFrom(const vector<char>& s) {
        sel = s; fill(rcount.begin(), rcount.end(), 0); cost = 0; benefit = 0;
        for (int i = 0; i < I->m; ++i) if (sel[i]) {
            for (int j : I->itemRes[i]) { if (rcount[j] == 0) cost += I->w[j]; rcount[j]++; }
            benefit += I->p[i];
        }
    }
};

// -----------------------------------------------------------------------------
//  (1) GREEDY DETERMINISTA
//      En cada paso agrega la alternativa factible con mejor razon
//      score = p_i / costo_marginal_i  (las "gratis" -> prioridad maxima).
//      Empates: menor indice. Siempre entrega la misma solucion.
// -----------------------------------------------------------------------------
State greedyDeterministic(const Instance& I) {
    State s; s.init(&I);
    while (true) {
        int best = -1; double bestScore = -1.0;
        for (int i = 0; i < I.m; ++i) if (!s.sel[i]) {
            ll mc = s.marginalAdd(i);
            if (s.cost + mc > I.B) continue;               // no factible
            double score = (mc == 0) ? 1e18 + (double)I.p[i]
                                     : (double)I.p[i] / (double)mc;
            if (score > bestScore) { bestScore = score; best = i; }
        }
        if (best < 0) break;
        s.add(best);
    }
    return s;
}

// -----------------------------------------------------------------------------
//  (2) GREEDY ESTOCASTICO  (GRASP)
//      Construye una Lista Restringida de Candidatos (RCL) con los items cuyo
//      score >= smax - alpha*(smax - smin) y elige uno al azar (semilla fija).
//      alpha=0 -> totalmente greedy ; alpha=1 -> totalmente aleatorio.
// -----------------------------------------------------------------------------
State greedyStochastic(const Instance& I, mt19937_64& rng, double alpha) {
    State s; s.init(&I);
    vector<pair<double,int>> cand;
    while (true) {
        cand.clear();
        double smin = 1e300, smax = -1e300;
        for (int i = 0; i < I.m; ++i) if (!s.sel[i]) {
            ll mc = s.marginalAdd(i);
            if (s.cost + mc > I.B) continue;
            double score = (mc == 0) ? 1e18 + (double)I.p[i]
                                     : (double)I.p[i] / (double)mc;
            cand.push_back({score, i});
            smin = min(smin, score); smax = max(smax, score);
        }
        if (cand.empty()) break;
        double thr = smax - alpha * (smax - smin);
        // construir RCL
        int cnt = 0; for (auto& c : cand) if (c.first >= thr) ++cnt;
        int pick = -1, idx = (int)(rng() % cnt);
        for (auto& c : cand) if (c.first >= thr) { if (idx-- == 0) { pick = c.second; break; } }
        s.add(pick);
    }
    return s;
}

// Orden estatico para el relleno: razon p_i / costo_propio (descendente)
vector<int> computeFillOrder(const Instance& I) {
    vector<int> order(I.m);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b){
        double ra = (double)I.p[a] / (double)max((ll)1, I.selfCost[a]);
        double rb = (double)I.p[b] / (double)max((ll)1, I.selfCost[b]);
        return ra > rb;
    });
    return order;
}

// Decodifica un cromosoma/sel a una solucion factible y localmente mejorada:
//   1) REPARACION: mientras infactible, quita el menos eficiente (torneo de 3)
//   2) RELLENO   : agrega items factibles segun orden estatico de razon
// Deja en s la solucion decodificada y devuelve su beneficio.
ll repairAndFill(State& s, const vector<char>& chromo,
                 const vector<int>& fillOrder, mt19937_64& rng) {
    s.setFrom(chromo);
    while (s.cost > s.I->B) {                       // reparacion
        int rem = -1; double bestEff = 1e300;
        for (int t = 0; t < 3; ++t) {
            int k = (int)(rng() % s.I->m);
            if (!s.sel[k]) continue;
            ll fr = s.freedRemove(k);
            double eff = (double)s.I->p[k] / (double)max((ll)1, fr);
            if (eff < bestEff) { bestEff = eff; rem = k; }
        }
        if (rem < 0) { for (int k = 0; k < s.I->m; ++k) if (s.sel[k]) { rem = k; break; } }
        if (rem < 0) break;
        s.remove(rem);
    }
    for (int i : fillOrder) if (!s.sel[i]) {        // relleno
        ll mc = s.marginalAdd(i);
        if (s.cost + mc <= s.I->B) s.add(i);
    }
    return s.benefit;
}

// -----------------------------------------------------------------------------
//  (3) SIMULATED ANNEALING  (sobre un decodificador con reparacion + relleno)
//      - Estado: vector binario, siempre decodificado a una solucion factible
//        y "rellenada" (optimo local de relleno).
//      - Movimiento: invertir 1 bit al azar y re-decodificar.
//      - Aceptacion de Metropolis sobre la FO decodificada.
//      - Temperatura inicial calibrada (Ben-Ameur):  T0 = -mean|dE| / ln(chi0),
//        con chi0 = probabilidad de aceptacion inicial deseada.
//      - Enfriamiento geometrico:  T <- alpha * T  cada L movimientos.
// -----------------------------------------------------------------------------
struct SAResult {
    ll best = 0;
    vector<char> bestSel;
    double timeMs = 0;
    long long moves = 0;
    long long bestMove = 0;                 // movimiento donde se hallo el optimo
    vector<pair<long long,ll>> trace;       // (movimiento, mejor FO)
    double T0 = 0;
};

SAResult simulatedAnnealing(const Instance& I, const vector<char>& startSel,
                            const vector<int>& fillOrder,
                            uint64_t seed, double alpha, double chi0,
                            long long maxMoves, bool recordTrace) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);
    State scratch; scratch.init(&I);
    SAResult R;

    // --- solucion inicial decodificada
    ll curBen = repairAndFill(scratch, startSel, fillOrder, rng);
    vector<char> curSel = scratch.sel;
    R.best = curBen; R.bestSel = curSel;

    // --- calibracion de T0 muestreando movimientos al azar
    double accDE = 0; int nDE = 0;
    for (int t = 0; t < 200; ++t) {
        vector<char> cand = curSel;
        cand[rng() % I.m] ^= 1;
        ll b = repairAndFill(scratch, cand, fillOrder, rng);
        double d = fabs((double)curBen - (double)b);
        if (d > 0) { accDE += d; ++nDE; }
    }
    double meanAbsDE = (nDE > 0) ? accDE / nDE : 1.0;
    double T0 = -meanAbsDE / log(chi0);
    if (!(T0 > 0)) T0 = meanAbsDE > 0 ? meanAbsDE : 1.0;
    R.T0 = T0;
    double T = T0;
    long long L = max<long long>(20, I.m / 2);     // largo de cadena por temperatura

    auto t0 = chrono::high_resolution_clock::now();
    long long sinceCool = 0;
    long long recordEvery = max<long long>(1, maxMoves / 500);

    for (long long mv = 1; mv <= maxMoves; ++mv) {
        vector<char> cand = curSel;
        cand[rng() % I.m] ^= 1;                      // invertir 1 bit
        ll b = repairAndFill(scratch, cand, fillOrder, rng);
        double dE = (double)curBen - (double)b;      // minimizamos -beneficio
        if (dE <= 0 || U(rng) < exp(-dE / T)) {
            curSel = scratch.sel; curBen = b;
            if (b > R.best) { R.best = b; R.bestSel = curSel; R.bestMove = mv; }
        }
        if (++sinceCool >= L) { T *= alpha; sinceCool = 0; if (T < 1e-9) T = 1e-9; }
        if (recordTrace && (mv % recordEvery == 0)) R.trace.push_back({mv, R.best});
    }
    auto t1 = chrono::high_resolution_clock::now();
    R.timeMs = chrono::duration<double, milli>(t1 - t0).count();
    R.moves = maxMoves;
    if (recordTrace) R.trace.push_back({maxMoves, R.best});
    return R;
}

// -----------------------------------------------------------------------------
//  (4) ALGORITMO GENETICO 
//      - Cromosoma binario (alternativas seleccionadas).
//      - Poblacion inicial: ejecuciones de greedy estocastico (+ algunos al azar).
//      - Seleccion por torneo binario, cruce uniforme, mutacion bit-flip.
//      - Elitismo.
// -----------------------------------------------------------------------------
struct GAResult {
    ll best = 0;
    vector<char> bestSel;
    double timeMs = 0;
    vector<tuple<int,ll,double>> trace;   // (generacion, mejor FO factible, fitness promedio)
};

struct EvalGA {
    ll benefit = 0;
    ll cost = 0;
    bool feasible = false;
    double fitness = 0.0;
};

// Evalua directamente un cromosoma sin repararlo ni rellenarlo.
// Si es factible, el fitness es el beneficio.
// Si es infactible, se penaliza segun el exceso de capacidad.
EvalGA evaluateChromosomeClassic(const Instance& I,
                                 const vector<char>& chromo,
                                 double lambda) {
    EvalGA e;
    vector<char> usedResource(I.n, 0);

    for (int i = 0; i < I.m; ++i) {
        if (!chromo[i]) continue;

        e.benefit += I.p[i];

        for (int j : I.itemRes[i]) {
            if (!usedResource[j]) {
                usedResource[j] = 1;
                e.cost += I.w[j];
            }
        }
    }

    e.feasible = (e.cost <= I.B);

    if (e.feasible) {
        e.fitness = (double)e.benefit;
    } else {
        ll excess = e.cost - I.B;
        e.fitness = (double)e.benefit - lambda * (double)excess;
    }

    return e;
}

GAResult geneticAlgorithm(const Instance& I, uint64_t seed,
                          int popSize, int generations,
                          double pc, double pm, bool recordTrace) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> U(0.0, 1.0);

    vector<vector<char>> pop(popSize);
    vector<double> fit(popSize);

    GAResult R;

    // Penalizacion suficientemente grande para castigar soluciones infactibles.
    // totalP representa una cota superior simple del beneficio posible.
    ll totalP = 0;
    for (ll x : I.p) totalP += x;
    double lambda = (double)totalP + 1.0;

    // -------------------------------------------------------------------------
    // Poblacion inicial:
    // Se construye usando distintas ejecuciones del greedy estocastico.
    // -------------------------------------------------------------------------
    for (int t = 0; t < popSize; ++t) {
        double alpha = 0.1 + 0.8 * U(rng);   // distintos niveles de aleatoriedad
        State g = greedyStochastic(I, rng, alpha);

        pop[t] = g.sel;

        EvalGA e = evaluateChromosomeClassic(I, pop[t], lambda);
        fit[t] = e.fitness;

        if (e.feasible && e.benefit > R.best) {
            R.best = e.benefit;
            R.bestSel = pop[t];
        }
    }

    // Como la poblacion inicial viene de greedy estocastico, deberia existir
    // al menos una solucion factible. Esta validacion es solo preventiva.
    if (R.bestSel.empty()) {
        State g = greedyDeterministic(I);
        R.best = g.benefit;
        R.bestSel = g.sel;
    }

    auto t0 = chrono::high_resolution_clock::now();

    // Seleccion por torneo binario usando fitness penalizado.
    auto tournament = [&]() -> int {
        int a = (int)(rng() % popSize);
        int b = (int)(rng() % popSize);
        return (fit[a] >= fit[b]) ? a : b;
    };

    for (int gen = 1; gen <= generations; ++gen) {
        vector<vector<char>> npop(popSize);
        vector<double> nfit(popSize);

        // ---------------------------------------------------------------------
        // Elitismo:
        // Se conserva la mejor solucion factible encontrada hasta ahora.
        // ---------------------------------------------------------------------
        npop[0] = R.bestSel;
        EvalGA eliteEval = evaluateChromosomeClassic(I, npop[0], lambda);
        nfit[0] = eliteEval.fitness;

        for (int c = 1; c < popSize; ++c) {
            int pa = tournament();
            int pb = tournament();

            vector<char> child(I.m, 0);

            // -----------------------------------------------------------------
            // Cruce uniforme:
            // Cada gen del hijo se toma desde uno de los dos padres.
            // -----------------------------------------------------------------
            if (U(rng) < pc) {
                for (int i = 0; i < I.m; ++i) {
                    child[i] = (U(rng) < 0.5) ? pop[pa][i] : pop[pb][i];
                }
            } else {
                child = pop[pa];
            }

            // -----------------------------------------------------------------
            // Mutacion bit-flip:
            // Cada gen cambia con probabilidad pm.
            // -----------------------------------------------------------------
            for (int i = 0; i < I.m; ++i) {
                if (U(rng) < pm) {
                    child[i] ^= 1;
                }
            }

            // -----------------------------------------------------------------
            // Evaluacion directa del cromosoma sin reparacion ni relleno.
            // Si es infactible, recibe penalizacion.
            // -----------------------------------------------------------------
            EvalGA e = evaluateChromosomeClassic(I, child, lambda);

            npop[c] = child;
            nfit[c] = e.fitness;

            // El mejor reportado debe ser factible.
            if (e.feasible && e.benefit > R.best) {
                R.best = e.benefit;
                R.bestSel = child;
            }
        }

        pop.swap(npop);
        fit.swap(nfit);

        if (recordTrace) {
            double avgFit = 0.0;
            for (double f : fit) avgFit += f;
            avgFit /= popSize;

            R.trace.push_back({gen, R.best, avgFit});
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    R.timeMs = chrono::duration<double, milli>(t1 - t0).count();

    return R;
                          }
// -----------------------------------------------------------------------------
//  Utilidades de E/S
// -----------------------------------------------------------------------------
static double meanV(const vector<double>& v){ double s=0; for(double x:v)s+=x; return v.empty()?0:s/v.size(); }
static double stdV (const vector<double>& v){ if(v.size()<2)return 0; double mu=meanV(v),s=0; for(double x:v)s+=(x-mu)*(x-mu); return sqrt(s/(v.size()-1)); }

// -----------------------------------------------------------------------------
//  Bateria completa para una instancia
// -----------------------------------------------------------------------------
void runInstance(const Instance& I, const string& outDir, int RUNS) {
    vector<int> fillOrder = computeFillOrder(I);
    long long saMoves = min<long long>(80000, max<long long>(15000, 200LL * I.m));
    int popSize    = (I.m <= 200) ? 50  : 40;
    int gens       = (I.m <= 200) ? 250 : 150;
    double pc = 0.9, pm = 1.0 / I.m, chi0 = 0.85, alpha = 0.95;

    // ---- summary.csv (append) ----
    ofstream sum(outDir + "/summary_" + I.name + ".csv");
    sum << "instance,algorithm,start,run,seed,FO,feasible,time_ms,iters,best_iter\n";

    // (1) Greedy determinista
    {
        auto t0 = chrono::high_resolution_clock::now();
        State g = greedyDeterministic(I);
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double,milli>(t1-t0).count();
        sum << I.name << ",greedy_det,-,1,0," << g.benefit << "," << g.feasible()
            << "," << ms << ",0,0\n";
    }

    // (2) Greedy estocastico (RUNS ejecuciones, semillas 1..RUNS)
    State bestStochState; bestStochState.init(&I); ll bestStochFO = -1;
    State detState = greedyDeterministic(I);
    {
        for (int r = 1; r <= RUNS; ++r) {
            mt19937_64 rng(r);
            auto t0 = chrono::high_resolution_clock::now();
            State g = greedyStochastic(I, rng, 0.3);
            auto t1 = chrono::high_resolution_clock::now();
            double ms = chrono::duration<double,milli>(t1-t0).count();
            sum << I.name << ",greedy_stoch,-," << r << "," << r << "," << g.benefit
                << "," << g.feasible() << "," << ms << ",0,0\n";
            if (g.benefit > bestStochFO) { bestStochFO = g.benefit; bestStochState = g; }
        }
    }

    // (3) Simulated Annealing  -- desde greedy determinista  y  desde greedy estocastico
    auto runSA = [&](const State& start, const string& tag, bool trace) {
        ll bestFO = -1; vector<pair<long long,ll>> bestTrace;
        for (int r = 1; r <= RUNS; ++r) {
            bool rec = trace && (r == 1);
            SAResult R = simulatedAnnealing(I, start.sel, fillOrder, (uint64_t)(1000*I.m + r),
                                            alpha, chi0, saMoves, rec);
            sum << I.name << ",sa," << tag << "," << r << "," << r << "," << R.best
                << ",1," << R.timeMs << "," << R.moves << "," << R.bestMove << "\n";
            if (R.best > bestFO) { bestFO = R.best; }
            if (rec) bestTrace = R.trace;
        }
        if (trace) {
            ofstream tr(outDir + "/conv_sa_" + tag + "_" + I.name + ".csv");
            tr << "iter,best_FO\n";
            for (auto& pr : bestTrace) tr << pr.first << "," << pr.second << "\n";
        }
        return bestFO;
    };
    ll saDetBest   = runSA(detState,        "det",   true);
    ll saStochBest = runSA(bestStochState,  "stoch", true);

    // (4) Algoritmo Genetico (RUNS ejecuciones)
    {
        ll bestFO = -1; vector<tuple<int,ll,double>> bestTrace;
        for (int r = 1; r <= RUNS; ++r) {
            bool rec = (r == 1);
            GAResult R = geneticAlgorithm(I, (uint64_t)(7000*I.m + r),
                                          popSize, gens, pc, pm, rec);
            sum << I.name << ",ga,greedy_stoch," << r << "," << r << "," << R.best
                << ",1," << R.timeMs << "," << gens << ",0\n";
            if (R.best > bestFO) bestFO = R.best;
            if (rec) bestTrace = R.trace;
        }
        ofstream tr(outDir + "/conv_ga_" + I.name + ".csv");
        tr << "gen,best_FO,avg_FO\n";
        for (auto& t : bestTrace)
            tr << get<0>(t) << "," << get<1>(t) << "," << get<2>(t) << "\n";
        (void)bestFO;
    }
    sum.close();
    cerr << "  [ok] " << I.name
         << "  (SA moves=" << saMoves << ", GA pop=" << popSize << " gens=" << gens << ")\n";
}

// -----------------------------------------------------------------------------
//  Sweeps de parametros (justificacion) sobre una instancia representativa
// -----------------------------------------------------------------------------
void runSweeps(const Instance& I, const string& outDir, int RUNS) {
    vector<int> fillOrder = computeFillOrder(I);
    State det = greedyDeterministic(I);
    long long saMoves = min<long long>(40000, max<long long>(6000, 100LL * I.m));

    // ---- SA: barrido de tasa de enfriamiento alpha ----
    {
        ofstream f(outDir + "/sweep_sa_alpha_" + I.name + ".csv");
        f << "alpha,mean_FO,std_FO,mean_time_ms\n";
        for (double a : {0.80, 0.90, 0.95, 0.99}) {
            vector<double> fos, ts;
            for (int r = 1; r <= RUNS; ++r) {
                SAResult R = simulatedAnnealing(I, det.sel, fillOrder, (uint64_t)(31*I.m + r),
                                                a, 0.85, saMoves, false);
                fos.push_back((double)R.best); ts.push_back(R.timeMs);
            }
            f << a << "," << meanV(fos) << "," << stdV(fos) << "," << meanV(ts) << "\n";
        }
    }
    // ---- SA: barrido de temperatura inicial (chi0 = prob. aceptacion inicial) ----
    {
        ofstream f(outDir + "/sweep_sa_chi0_" + I.name + ".csv");
        f << "chi0,mean_FO,std_FO,mean_time_ms\n";
        for (double c : {0.50, 0.70, 0.85, 0.95}) {
            vector<double> fos, ts;
            for (int r = 1; r <= RUNS; ++r) {
                SAResult R = simulatedAnnealing(I, det.sel, fillOrder, (uint64_t)(53*I.m + r),
                                                0.95, c, saMoves, false);
                fos.push_back((double)R.best); ts.push_back(R.timeMs);
            }
            f << c << "," << meanV(fos) << "," << stdV(fos) << "," << meanV(ts) << "\n";
        }
    }
    // ---- GA: barrido de tamano de poblacion ----
    {
        ofstream f(outDir + "/sweep_ga_pop_" + I.name + ".csv");
        f << "pop,mean_FO,std_FO,mean_time_ms\n";
        for (int pop : {20, 50, 100}) {
            vector<double> fos, ts;
            for (int r = 1; r <= RUNS; ++r) {
                GAResult R = geneticAlgorithm(I, (uint64_t)(91*I.m + r),
                                              pop, 200, 0.9, 1.0/I.m, false);
                fos.push_back((double)R.best); ts.push_back(R.timeMs);
            }
            f << pop << "," << meanV(fos) << "," << stdV(fos) << "," << meanV(ts) << "\n";
        }
    }
    // ---- GA: barrido de tasa de mutacion ----
    {
        ofstream f(outDir + "/sweep_ga_pm_" + I.name + ".csv");
        f << "pm_x_m,mean_FO,std_FO,mean_time_ms\n";
        for (double k : {0.5, 1.0, 2.0, 4.0}) {
            vector<double> fos, ts;
            for (int r = 1; r <= RUNS; ++r) {
                GAResult R = geneticAlgorithm(I, (uint64_t)(97*I.m + r),
                                              50, 200, 0.9, k/I.m, false);
                fos.push_back((double)R.best); ts.push_back(R.timeMs);
            }
            f << k << "," << meanV(fos) << "," << stdV(fos) << "," << meanV(ts) << "\n";
        }
    }
    cerr << "  [ok] sweeps en " << I.name << "\n";
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    if (argc < 4) {
        cerr << "uso: " << argv[0] << " <archivo> <nombre> <outdir> [RUNS] [--sweep]\n";
        return 1;
    }
    string file = argv[1], name = argv[2], outDir = argv[3];
    int RUNS = 10;
    bool sweep = false;
    for (int a = 4; a < argc; ++a) {
        string s = argv[a];
        if (s == "--sweep") sweep = true;
        else RUNS = atoi(s.c_str());
    }
    Instance I = readInstance(file, name);
    cerr << "Instancia " << name << ": m=" << I.m << " n=" << I.n
         << " B=" << I.B << " (RUNS=" << RUNS << ")\n";
    if (sweep) runSweeps(I, outDir, RUNS);
    else       runInstance(I, outDir, RUNS);
    return 0;
}
