import numpy as np
import os

from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from tkinter import Toplevel
from gibbs_functions import first_SGS, calc_of_weights, gibbs_sampler, backtr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure





file_name = ""
var_model = ""
var_sill = ""
nugget_fn = ""
p_table_fn = ""
s_table_fn = ""
running = True
model = []
cc = []
nugget = []
arr = []



class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.init_window()

    def init_window(self):
        # splash = Splash(self)
        self.data_frame = Frame(self, bd=2, relief=RAISED, width=300, height=250)
        self.data_frame.grid_propagate(0)
        self.postproc_frame = Frame(self,bd=2, relief=RAISED, width=340, height=520)
        self.postproc_frame.grid_propagate(0)
        self.proc_frame = Frame(self, bd=2, relief=RAISED, width=300, height=20)
        self.proc_frame.grid_propagate(0)

        self.master.title("Co-Gibbs Sampler")
        self.pack(fill=BOTH, expand=1)
        self.master.iconbitmap(os.getcwd() + "\Gibbs.ico")

        self.lbHead = Label(self.data_frame, text="Input data",font='Helvetica 10 bold')
        self.lbHead.grid(row=1, column=0)
        self.lbHeadComp = Label(self.proc_frame, text="Computation progress", font='Helvetica 10 bold')
        self.lbHeadComp.grid(row=0, column=0, columnspan=2)
        self.lbHeadRes = Label(self.postproc_frame, text="Results", font='Helvetica 10 bold')
        self.lbHeadRes.grid(row=0, column=0)
        self.lbFilename = Label(self.data_frame, text = "Filename:")
        self.lbFilename.grid(row = 3, column = 0, sticky=E)
        self.lbFilename1 = Label(self.data_frame, text=file_name)
        self.lbFilename1.grid(row=3, column=1, sticky=W)
        self.lbVar_model = Label(self.data_frame, text="Variogram model:")
        self.lbVar_model.grid(row=4, column=0, sticky=E)
        self.lbVar_model1 = Label(self.data_frame, text=var_model)
        self.lbVar_model1.grid(row=4, column=1, sticky=W)
        self.lbVar_sill = Label(self.data_frame, text="Variogram sill:")
        self.lbVar_sill.grid(row=5, column=0, sticky=E)
        self.lbVar_sill1 = Label(self.data_frame, text=var_sill)
        self.lbVar_sill1.grid(row=5, column=1, sticky=W)
        self.lbVar_nugget = Label(self.data_frame, text="Nugget:")
        self.lbVar_nugget.grid(row=6, column=0, sticky=E)
        self.lbVar_nugget1 = Label(self.data_frame, text=nugget_fn)
        self.lbVar_nugget1.grid(row=6, column=1, sticky=W)
        self.lbBacktrP = Label(self.data_frame, text="Backtransformation table\n Principal variable:")
        self.lbBacktrP.grid(row=7, column=0, sticky=E)
        self.lbBacktrP1 = Label(self.data_frame, text=p_table_fn)
        self.lbBacktrP1.grid(row=7, column=1, sticky=W)
        self.lbBacktrS = Label(self.data_frame, text="Backtransformation table\n Auxiliary variable:")
        self.lbBacktrS.grid(row=8, column=0, sticky=E)
        self.lbBacktrS1 = Label(self.data_frame, text=s_table_fn)
        self.lbBacktrS1.grid(row=8, column=1, sticky=W)
        self.lbNreal = Label(self.data_frame, text="Number of Realizations:")
        self.lbNreal.grid(row=10, column=0, sticky=E)
        self.lbNiter = Label(self.data_frame, text="Number of Iterations \nfor Gibbs Sampler:")
        self.lbNiter.grid(row=20, column=0, sticky=E)
        self.lbIndP = Label(self.data_frame, text="Principal variable, Col No.:")
        self.lbIndP.grid(row=30, column=0, sticky=E)
        self.lbIndS = Label(self.data_frame, text="Auxiliary variable, Col No.:")
        self.lbIndS.grid(row=40, column=0, sticky=E)
        self.lbKrigType = Label(self.data_frame, text="Type of Kriging system:")
        self.lbKrigType.grid(row=60, column=0, sticky=E)
        self.lbCPU = Label(self.data_frame, text="CPU number:")
        self.lbCPU.grid(row=65, column=0, sticky=E)

        self.lbSGS = Label(self.proc_frame, text="First iteration with SGS")
        self.lbSGS.grid(row=1, column=0, sticky=E, padx=10)
        self.lbSGSrun = Label(self.proc_frame, text="Not started")
        self.lbSGSrun.grid(row=1, column=1, sticky=W, padx=10)
        self.lbWeights = Label(self.proc_frame, text="Calculation of Weights")
        self.lbWeights.grid(row=2, column=0, sticky=E, padx=10)
        self.lbWeightsrun = Label(self.proc_frame, text="Not started")
        self.lbWeightsrun.grid(row=2, column=1, sticky=W, padx=10)
        self.lbGibbs = Label(self.proc_frame, text="Gibbs sampler")
        self.lbGibbs.grid(row=3, column=0, sticky=E, padx=10)
        self.lbGibbsrun = Label(self.proc_frame, text="Not started")
        self.lbGibbsrun.grid(row=3, column=1, sticky=W, padx=10)


        ttk.Separator(self.proc_frame, orient=HORIZONTAL).place(x=0, y=110, relwidth=1)
        self.lbGibbs = Label(self.proc_frame, text=r"NU, School of Mining and Geosciences",font='Helvetica 10 italic').place(x=0,y=115)

        self.lenvar = IntVar()
        self.lenvar = 100
        self.enNreal = Entry(self.data_frame)
        self.enNreal.insert(END, str(self.lenvar))
        self.enNreal.grid(row=10, column=1, sticky=E, padx=10)
        self.enNrealGibbs = Entry(self.data_frame)
        self.enNrealGibbs.insert(END, "200")
        self.enNrealGibbs.grid(row=20, column=1, sticky=E, padx=10)
        self.enPrimaryColInd = Entry(self.data_frame)
        self.enPrimaryColInd.grid(row=30, column=1, sticky=E, padx=10)
        self.enSecondColInd = Entry(self.data_frame)
        self.enSecondColInd.grid(row=40, column=1, sticky=E, padx=10)
        self.enCPU = Entry(self.data_frame)
        self.enCPU.insert(END, "-1")
        self.enCPU.grid(row=65, column=1, sticky=E, padx=10)

        self.lbRealN = Label(self.postproc_frame, text="Realization No.")
        self.lbRealN.grid(row=1, column=0, sticky=SE, padx=10)

        self.enSim1Ind = Entry(self.postproc_frame)
        self.enSim1Ind.grid(row=1, column=1, padx=10)

        self.scSim1Ind = Scale(self.postproc_frame, from_=1, to_=self.lenvar, orient=HORIZONTAL, length=200, command=self.onScale)
        self.scSim1Ind.grid(row=1, column=1, sticky=N)
        self.var = IntVar()


        self.KrigType = ttk.Combobox(self.data_frame, width=17)
        self.KrigType['values'] = (
            "Multicollocated cokriging", "Collocated cokriging", "Simple cokriging", "Simple kriging")
        self.KrigType.current(0)
        self.KrigType.grid(row=60, column=1)

        self.runbtn = Button(self.data_frame, text="Run", command=self.rungibbs).grid(row=90, column=1)

        self.pltbtn = Button(self.postproc_frame, text="Plot results", command=self.plotfn).grid(row=2, column=1)
        self.updtbtn = Button(self.postproc_frame, text="Update scale", command=self.updateScale).grid(row=0, column=1)

        self.lbCorr = Label(self.postproc_frame, text="Correlation:")
        self.lbCorr.grid(row=3, column=0, sticky=E)


        self.data_frame.grid(row=0, column=0,padx=1,sticky=N+W+E+S)
        self.postproc_frame.grid(row=0, column=1,padx=1,rowspan = 2,sticky=(N,E,W,S))
        self.proc_frame.grid(row=1, column=0, padx=1, sticky=(N,S,W,E))


        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu, tearoff=False)
        file.add_command(label='Import heterotopic dataset', command=self.load_file)
        file.add_command(label='Import variogram model', command=self.load_model)
        file.add_command(label='Import variogram sill', command=self.load_sill)
        file.add_command(label='Import variogram nugget', command=self.load_nugget)
        submenu = Menu(menu, tearoff=False)
        submenu.add_command(label='Principal variable', command=self.load_p_transtable)
        submenu.add_command(label='Auxiliary variable', command=self.load_s_transtable)
        file.add_cascade(label='Import transformation table', menu=submenu, underline=0)
        file.add_separator()
        file.add_command(label='Export results', command=self.export)
        file.add_separator()
        file.add_command(label='Exit', command=root.quit)
        menu.add_cascade(label='File', menu=file)


    def onScale(self, val):

        v = int(float(val))
        self.var.set(v)

    def updateScale(self):
        self.lenvar = int(self.enNreal.get())
        self.scSim1Ind['to_'] = self.lenvar


    def plotfn(self):

        try:
            simul_1_b, simul_2_b
        except NameError:
            messagebox.showerror("Error", "Not enough data! Enter Index value or run the Gibbs sampler!")
        else:
            ind_1 = int(self.scSim1Ind.get())-1
            corr1 = np.corrcoef(simul_1[:,ind_1], simul_2[:,1])
            corr = np.zeros([simul_1.shape[1],1])
            for i in range(simul_1_b.shape[1]):
                corr[i] = np.corrcoef(simul_1[:,i], simul_2[:,1])[0,1]

            self.lbCorrVal = Label(self.postproc_frame, text=round(corr1[0,1],4))
            self.lbCorrVal.grid(row=3, column=1, sticky=W)
            f = Figure(figsize=(3,4), dpi=100)
            a = f.add_subplot(211)
            a.scatter(simul_1_b[:,ind_1], simul_2_b[:,1], s=5)
            a.set_xlabel("Principal Variable")
            a.set_ylabel("Auxiliary Variable")
            a.set_title("Scatterplot")
            b = f.add_subplot(212)
            b.hist(corr, bins=20)
            b.set_xlabel("Mean Correlation")
            b.set_ylabel("Frequency")
            f.tight_layout()

            canvas = FigureCanvasTkAgg(f, self.postproc_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row = 4, column = 0, columnspan=2, padx=20)



    def export(self):
        try:
            simul_1_b
        except NameError:
            messagebox.showerror("Error",
                                 "You haven't run the Gibbs sampler, you have to run it before showing the results!")
        else:
            exp_file_name = filedialog.asksaveasfilename(filetypes=(
            ("TXT files", "*.txt"), ("CSV files", "*.csv"), ("OUT files", "*.out"), ("ALL files", "*.*")))

            np.savetxt(exp_file_name, simul_1_b)

    def rungibbs(self):

        global simul, simul_1, simul_2, simul_1_b, simul_2_b

        if (len(arr) == 0):
            messagebox.showerror("Error", "You didn't import the dataset! Please import before beginning!")
        else:
            nreal = int(self.enNreal.get())
            niter = int(self.enNrealGibbs.get())
            krigtype = self.KrigType.get()
            ind_p = int(self.enPrimaryColInd.get())
            ind_s = int(self.enSecondColInd.get())
            cpuNumber = int(self.enCPU.get())
            coord = arr[:, 0:3]
            data = arr[:, [ind_p - 1, ind_s - 1]]
            ndata = len(data)
            index = np.transpose(np.array(np.where(data[:, 0] == -99)))
            index_l = np.where(data[:, 1] != -99)
            coord_new = np.delete(coord, index, axis=0)
            data_new = np.delete(data, index, axis=0)

            if krigtype == "Simple kriging":
                cc[:,1:3] = 0

            self.lbSGSrun['text'] = "running"

            self.update()

            [simul, simul_1, simul_2] = first_SGS(cpuNumber, index, nreal, coord_new, coord, data_new, data, model, cc,
                                                  nugget, krigtype)
            self.lbSGSrun['text'] = "Completed"
            self.lbWeightsrun['text'] = "running..."

            self.update()


            [weights_final, prediction_var] = calc_of_weights(cpuNumber, data, index, coord, model, cc, nugget,
                                                              krigtype)

            self.lbWeightsrun['text'] = "Completed"
            self.lbGibbsrun['text'] = "running..."
            self.update()

            # Gibbs Sampler
            self.Gibbsprogbar = ttk.Progressbar(self.proc_frame, orient="horizontal", length=120, mode="determinate")
            self.Gibbsprogbar.grid(row=4, column=1)
            self.Gibbsprogbar['maximum'] = niter - 1

            simul_1 = gibbs_sampler(self, index, simul_1, simul_2, data, niter, nreal, weights_final, prediction_var,
                                    krigtype)
            self.lbGibbsrun['text'] = "Completed"
            self.lbGibbsrun.update()
            simul_1_b = backtr(simul_1, p_table, min(p_table[:, 0]), max(p_table[:, 0]), np.array([1, 1]))
            simul_2_b = backtr(simul_2, s_table, min(s_table[:, 0]), max(s_table[:, 0]), np.array([1, 1]))



    def create_element(self, types, labeltext, row, column, pos):
        elementname = types(self, text=labeltext)
        elementname.grid(row=row, column=column, sticky=pos)


    def load_model(self):
        self.master.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select A Model File",
                                                          filetypes=[("model files", "*.mod")])
        global model, var_model

        fpath = self.master.filename
        if (os.path.isfile(fpath)):
            with open(fpath) as fp:
                arr = np.loadtxt(fp, dtype=float)
            fp.close()
        model = arr
        var_model = os.path.basename(fpath)
        self.lbVar_model1['text'] = var_model

    def load_sill(self):
        self.master.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select A Sill File",
                                                          filetypes=[("sill files", "*.cc")])
        global cc, var_sill
        fpath = self.master.filename
        if (os.path.isfile(fpath)):
            with open(fpath) as fp:
                arr = np.loadtxt(fp, dtype=float)
            fp.close()
        cc = arr
        var_sill = os.path.basename(fpath)
        self.lbVar_sill1['text'] = var_sill


    def load_nugget(self):
        self.master.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select A Nugget File",
                                                          filetypes=[("nugget files", "*.nug")])
        global nugget, nugget_fn

        fpath = self.master.filename
        if (os.path.isfile(fpath)):
            with open(fpath) as fp:
                arr = np.loadtxt(fp, dtype=float)
            fp.close()

        nugget = arr
        nugget_fn = os.path.basename(fpath)
        self.lbVar_nugget1['text'] = nugget_fn

    def load_s_transtable(self):
        self.master.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Secondary variable Transformation table File",
                                                          filetypes=[("table files", "*.trn")])
        global s_table, s_table_fn

        fpath = self.master.filename
        if (os.path.isfile(fpath)):
            with open(fpath) as fp:
                arr = np.loadtxt(fp, dtype=float)
            fp.close()
        s_table = arr
        s_table_fn = os.path.basename(fpath)
        self.lbBacktrS1['text'] = s_table_fn


    def load_p_transtable(self):
        self.master.filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                                          title="Primary variable Transformation table File",
                                                          filetypes=[("table files", "*.trn")])
        global p_table, p_table_fn

        fpath = self.master.filename
        if (os.path.isfile(fpath)):
            with open(fpath) as fp:
                arr = np.loadtxt(fp, dtype=float)
            fp.close()
        p_table = arr
        p_table_fn = os.path.basename(fpath)
        self.lbBacktrP1['text'] = p_table_fn

    def load_file(self):
        self.master.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select A File",
                                                          filetypes=[("out files", "*.out")])
        filepath = self.master.filename
        global file_name
        file_name = os.path.basename(filepath)
        global colNames
        colNames = []

        if (os.path.isfile(filepath)):
            with open(filepath) as fp:
                global filetitle
                global arr
                global colNumber
                filetitle = fp.readline()[:-1]
                colNumber = int(fp.readline()[:-1])
                for i in range(colNumber):
                    colNames.append(fp.readline()[:-1])
                arr = []
                for line in fp:
                    temp = np.array(line[:-1].split())
                    temp = temp.transpose()
                    arr.append(temp)
                arr = np.array(arr)
                arr = arr.astype(np.float)
            self.lbFilename1['text'] = file_name

            fp.close()



if __name__ == "__main__":
    root = Tk()
    root.geometry("650x540")
    root.resizable(0,0)

    app = Window(root)

    root.mainloop()

