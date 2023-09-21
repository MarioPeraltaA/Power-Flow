"""Power Flow study.

This program runs a power flow study for arbitrary
Transmission network.

    Author: Mario Roberto Peralta A.
    email: Mario.Peralta@ucr.ac.cr

"""

import numpy as np
import pandas as pd

class sysData():
    def __init__(self):
        self._basis = {}
        self._buses = {}
        self._lines = {}
        self._transformers = {}
        self._loads = {}
        self._capacitors = {}
        self._generators = {}

    def call_data(self, path: str) -> None:
        # Read data
        df = pd.read_excel(path,
                           sheet_name=None)
        # List of all sheets
        sheets = list(df.keys())

        # Set data regarding the sheet
        for sheet in df.keys():
            # Basis
            if sheet == sheets[0]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self._basis[c] = cols
            # Buses
            if sheet == sheets[1]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self._buses[c] = cols
            # Lines
            elif sheet == sheets[2]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self._lines[c] = cols
            # Transformers
            elif sheet == sheets[3]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self._transformers[c] = cols
            # Loads
            elif sheet == sheets[4]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self._loads[c] = cols
            # Capacitors
            elif sheet == sheets[5]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self._capacitors[c] = cols
            # Generators
            elif sheet == sheets[6]:
                for c in df[sheet].columns:
                    cols = [v for v in df[sheet][c]]
                    # Update attribute
                    self._generators[c] = cols


class Bus():
    pass

    def __repr__(self) -> str:
        return f"<{self.Bus}>"


class Line():
    """Gegenal type of conductors.

    Represents fundamentally the pi model of a
    conductors or transformer.

    """
    pass

    def __repr__(self) -> str:
        return f"({self._from_bus}, {self._to_bus})"


class Transformer():
    """Model of transformer.

    n : 1.

    """
    pass


class Capacitor():
    pass


class Load():
    pass


class Generator():
    pass


class Slack():
    def __init__(self):
        self._V = 1
        self._phase = 0    # [Rad]
        self._PL = 0
        self._QL = 0

    def __repr__(self) -> str:
        return f"<{self.Bus}>"


class System():

    def __init__(self):
        self.Sb = None
        self._bus_labels = []
        self._lines = []
        self._transformers = []
        self._loads = []
        self._capacitors = []
        self._generators = []

    def set_basis(self, data: sysData) -> None:
        basis = data._basis
        self.Sb = basis["S (MVA)"][0]

    def add_bus_labels(self, data: sysData) -> None:
        buses = data._buses
        Nbuses = len(buses["Bus"])
        for b in range(Nbuses):
            bus = Bus()
            for ft, vals in buses.items():
                if " " in ft:
                    ft, _ = ft.split(" ")
                    setattr(bus, ft, vals[b])
                else:
                    setattr(bus, ft, vals[b])
            self._bus_labels.append(bus)

    def add_lines(self, data: sysData) -> None:
        lines = data._lines
        Nlines = len(lines["Bus1"])
        for b in range(Nlines):
            line = Line()
            for ft, vals in lines.items():
                if " " in ft:
                    ft, _ = ft.split(" ")
                    setattr(line, ft, vals[b])
                else:
                    setattr(line, ft, vals[b])
            self._lines.append(line)

    def add_transformers(self, data: sysData) -> None:
        txs = data._transformers
        Ntxs = len(txs["Bus1"])
        for t in range(Ntxs):
            tx = Transformer()
            for ft, vals in txs.items():
                if " " in ft:
                    fts = ft.split(" ")
                    ft = fts[0]
                    setattr(tx, ft, vals[t])
                else:
                    setattr(tx, ft, vals[t])
            self._transformers.append(tx)

    def add_loads(self, data: sysData) -> None:
        loads = data._loads
        Nloads = len(loads["Bus"])
        for L in range(Nloads):
            load = Load()
            for ft, vals in loads.items():
                if " " in ft:
                    ft, _ = ft.split(" ")
                    setattr(load, ft, vals[L])
                else:
                    setattr(load, ft, vals[L])
            self._loads.append(load)

    def add_null_PQbuses(self) -> None:
        """Null injections buses.

        It creates and adds :py:class:`Load` instances of
        null injection power.

        """
        # All load buses
        PQbuses = [load.Bus for load
                   in self._bus_labels if load.Code == "PQ"]
        # Other than null injection
        PQdata = [load.Bus for load in self._loads]
        # Null injection buses
        for bus in PQbuses:
            if bus not in PQdata:
                attrs = {"Bus": bus, "Pload": 0.0, "Qload": 0.0}
                nullPQload = Load()
                for ft, val in attrs.items():
                    setattr(nullPQload, ft, val)
                self._loads.append(nullPQload)

        return self._loads

    def add_capacitors(self, data: sysData) -> None:
        capas = data._capacitors
        Ncapas = len(capas["Bus"])
        for c in range(Ncapas):
            capa = Capacitor()
            for ft, vals in capas.items():
                if " " in ft:
                    ft, _ = ft.split(" ")
                    setattr(capa, ft, vals[c])
                else:
                    setattr(capa, ft, vals[c])
            self._capacitors.append(capa)

    def add_generators(self, data: sysData) -> None:
        gens = data._generators
        Ngens = len(gens["Bus"])
        for g in range(Ngens):
            gen = Generator()
            for ft, vals in gens.items():
                if " " in ft:
                    fts = ft.split(" ")
                    ft = fts[0]
                    setattr(gen, ft, vals[g])
                else:
                    setattr(gen, ft, vals[g])
            self._generators.append(gen)

    def set_buses(self) -> None:
        """Sort and store buses.

        It allocates "OSC", "PQ" or "PV" labels code
        regarding the equipment values the bus is
        connected to and sort them in this order:
        Slack bus, Load buses, Voltage-controlled buses.

        Creats attributes:
            :py:attr:`System._but_slack_buses`
            :py:attr:`System._PQ_buses`
            :py:attr:`System._PV_buses`

        """
        # Creat attribute
        self._buses = []
        # Retrieve slack bus label from data
        for b in self._bus_labels:
            if b.Code == "OSC":
                slack = Slack()
                slack.Bus = b.Bus     # Index label
                slack.Code = b.Code   # Type of bus
        self._buses.append(slack)

        # Creat null injection buses
        loads = self.add_null_PQbuses()
        # Retrieve PQbuses
        PQbuses = [b for b in self._bus_labels if b.Code == "PQ"]
        # Update attributes of buses
        for loadB in PQbuses:
            label = loadB.Bus
            for L in loads:
                if L.Bus == label:
                    # Add as pu
                    loadB._PL = - L.Pload / self.Sb
                    loadB._QL = - L.Qload / self.Sb
                    loadB._V = 1.0
                    loadB._phase = 0.0

        # Retrieve PVbuses
        PVbuses = [v for v in self._bus_labels if v.Code == "PV"]
        # Update attributes of buses
        for genB in PVbuses:
            label = genB.Bus
            for G in self._generators:
                if G.Bus == label:
                    genB._V = G.Vref
                    # Add as pu
                    genB._PL = G.PGen / self.Sb
                    genB._QL = G.QGen / self.Sb
                    genB._phase = 0.0

        # Sort out buses of the system
        self._buses += PQbuses
        self._buses += PVbuses

        # Capacitores
        for c in self._capacitors:
            label = c.Bus
            for b in self._buses:
                if b.Bus == label:
                    b._QL += c.B / self.Sb

        # Set new attributes
        PQs = [q for q in self._buses if q.Code == "PQ"]
        PVs = [g for g in self._buses if g.Code == "PV"]
        self._but_slack_buses = PQs + PVs
        self._PQ_buses = PQs
        self._PV_buses = PVs

    def from_to_buses(self):
        """Ends of lines and transformers.

        Update from_bus and to_bus attributes
        with actual Bus instances.

        """
        # Ends of lines
        for L in self._lines:
            from_label = L.Bus1
            to_label = L.Bus2
            for b in self._buses:
                if b.Bus == from_label:
                    from_bus = b
                    for p in self._buses:
                        if p.Bus == to_label:
                            to_bus = p
                            L._from_bus = from_bus
                            L._to_bus = to_bus
        # Ends of transformers
        for T in self._transformers:
            from_label = T.Bus1
            to_label = T.Bus2
            for b in self._buses:
                if b.Bus == from_label:
                    from_bus = b
                    for p in self._buses:
                        if p.Bus == to_label:
                            to_bus = p
                            T._from_bus = from_bus
                            T._to_bus = to_bus

    @property
    def build_Y(self) -> np.ndarray:
        nBuses = len(self._buses)
        Y = np.zeros((nBuses, nBuses), dtype=complex)
        for i, b in enumerate(self._buses):
            if hasattr(b, "B"):
                c_pu = 1j*b.B
                Y[i, i] += c_pu

        for line in self._lines:
            m = self._buses.index(line._from_bus)
            n = self._buses.index(line._to_bus)
            # Get series admitance of the line
            Y_serie = (1) / (line.R + 1j*line.X)
            from_L = 1j*line.B / 2
            to_L = 1j*line.B / 2
            # Build Y
            Y[m, m] += from_L + Y_serie
            Y[n, n] += to_L + Y_serie
            Y[m, n] -= Y_serie
            Y[n, m] -= Y_serie

        for tx in self._transformers:
            n = tx.Bus1_tap
            Zcc = tx.Rcc + 1j*tx.Xcc
            Ycc = 1 / Zcc
            Y_serie = n * Ycc
            from_T = (1-n)*Ycc
            to_T = (abs(n)**2 - n) * Ycc
            p = self._buses.index(tx._from_bus)
            q = self._buses.index(tx._to_bus)
            # Build Y
            Y[p, p] += from_T + Y_serie
            Y[q, q] += to_T + Y_serie
            Y[p, q] -= Y_serie
            Y[q, p] -= Y_serie
        self._Y = Y
        return Y

    @property
    def delta_F(self) -> np.ndarray:
        """Mismatch functions.

        Evaluates states given as attributes
        and returns the mismatch column vector
        of P and Q at bus k.

        """
        Y = self._Y
        # Calculated power
        Pcals = []
        Qcals = []
        # Schedule power
        Psch = []
        Qsch = []
        for k, busk in enumerate(self._buses):
            if busk.Code == "OSC":
                continue
            elif busk.Code == "PQ":
                Pcals.append(busk._PL)
                Qcals.append(busk._QL)
                summaPk = 0
                summaQk = 0
                for i, b in enumerate(self._buses):
                    # Pk
                    mag = abs(busk._V) * abs(b._V)
                    cosP = Y[k, i].real * np.cos(busk._phase - b._phase)
                    sinP = Y[k, i].imag * np.sin(busk._phase - b._phase)
                    summaPk += mag*(cosP + sinP)
                    # Qk
                    magQ = abs(busk._V) * abs(b._V)
                    sinQ = Y[k, i].real * np.sin(busk._phase - b._phase)
                    cosQ = Y[k, i].imag * np.cos(busk._phase - b._phase)
                    summaQk += magQ*(sinQ - cosQ)
                Psch.append(summaPk)
                Qsch.append(summaQk)
            elif busk.Code == "PV":
                Pcals.append(busk._PL)
                summaPk = 0
                for i, b in enumerate(self._buses):
                    # Pk
                    mag = abs(busk._V) * abs(b._V)
                    cosP = Y[k, i].real * np.cos(busk._phase - b._phase)
                    sinP = Y[k, i].imag * np.sin(busk._phase - b._phase)
                    summaPk += mag*(cosP + sinP)
                Psch.append(summaPk)

        # Mismatches
        fcal = np.array(Pcals + Qcals)
        fsch = np.array(Psch + Qsch)
        self._delta = fsch - fcal
        return self._delta

    def update_states(self, x) -> None:
        """Update attributes.

        Given a column vector `x` it sets such states
        as attributes of the respective kind of bus.

        """
        # V_angles: All but slack buses
        for i, bus in enumerate(self._but_slack_buses):
            bus._phase = x[i]

        # V_magnitudes: PQ buses only
        for n, bus in enumerate(self._PQ_buses):
            bus._V = x[len(self._but_slack_buses)+n]

    def jac(self,
            x: np.ndarray,
            h: float = 1e-6) -> np.ndarray:
        """"Jacobian Matrix.

        Approximate partial derivative through secant line.

        """
        # Initialize array
        J = np.empty((len(x), len(x)))
        self.update_states(x)
        f = self.delta_F     # States at previous x
        for var_ind in range(len(x)):
            dx = np.zeros(len(x))
            dx[var_ind] = h
            # For small h
            self.update_states(x + dx)
            fdelta = self.delta_F
            sec = (fdelta - f) / h
            J[:, var_ind] = sec
            self._delta = f
        self._J = J
        return J

    def newton(self,
               tol: float = 1e-3,
               max_iters: int = 10) -> np.ndarray:
        """Power flow.

        Runs a power flow study by Newton-Raphson iterative
        method. It returns `False` in case the method
        dit not converge.

        """
        # Flat-start
        x0 = len(self._but_slack_buses)*[0] + len(self._PQ_buses)*[1]
        x = np.array(x0, dtype=float)
        iters = 0
        # Initial conditions
        self.build_Y
        self.delta_F
        self.jac(x)

        while (max(np.abs(self._delta)) > tol) and iters < max_iters:
            try:
                # Newton-Raphson definition
                x -= np.matmul(np.linalg.inv(self._J), self._delta)
                iters += 1
                # Update states
                self.update_states(x)
                self.delta_F
                self.jac(x)
            except np.linalg.LinAlgError as e:
                print(f"{e}: No inverse possible to Jacobian.")
                return False
        if iters == max_iters:
            print(f"Newton-Raphson did not converged after {iters} iterations.")
            return None

        return x


if __name__ == "__main__":
    sysdata = sysData()
    directory = "./sys/IEEE_39_bus.xlsx"
    sysdata.call_data(directory)
    sys = System()
    sys.set_basis(sysdata)
    sys.add_bus_labels(sysdata)
    sys.add_lines(sysdata)
    sys.add_transformers(sysdata)
    sys.add_loads(sysdata)
    sys.add_capacitors(sysdata)
    sys.add_generators(sysdata)
    sys.set_buses()
    sys.from_to_buses()
    x = sys.newton()
