import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QWidget, QVBoxLayout,QHBoxLayout, QTabWidget # QProgressBar, QLineEdit
from PyQt5.QtGui import QDoubleValidator
from PyQt5.uic import loadUi
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
import my_functions as myfs
# import qtmodern.styles
# import qtmodern.windows
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph import PColorMeshItem, ScatterPlotItem

pen = pg.mkPen(color='r', width=3, style=QtCore.Qt.DashLine)


# plt.rcParams.update({'font.size':8})
#plt.rc('axes',edgecolor='w')
# if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
#     QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
# if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class OtherWindow(QWidget):
    ## Handles full map pop up
    def __init__(self, parent=None):
        super(OtherWindow, self).__init__(parent)
        self.figure = Figure(tight_layout=True)
        #self.figure.set_facecolor('gray')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111)
        #self.axis.set_facecolor('silver')
        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.canvas)


class Fit(QMainWindow):
    def __init__(self):
        super(Fit, self).__init__()
        loadUi("loadfile_layout_pyqtgraph.ui", self)
        # self.setFixedSize(2000, 2000)
        self.show()
        # self.showMaximized()
        self.filebutton.clicked.connect(self.file_function)
        self.row = 0
        self.col = 0
        # self.file_function()
        self.nextCol.clicked.connect(self.nextCol_function)
        self.prevCol.clicked.connect(self.prevCol_function)
        self.nextRow.clicked.connect(self.nextRow_function)
        self.prevRow.clicked.connect(self.prevRow_function)
        self.fitButton.clicked.connect(self.fit_function)
        self.reFitButton.clicked.connect(self.reFit_function)
        self.hideButton.clicked.connect(self.hideFunction)
        self.unhideButton.clicked.connect(self.unhideFunction)
        self.plotMap.clicked.connect(self.plotMapFun)
        self.nanButton.clicked.connect(self.nan_function)
        self.onlyDouble = QDoubleValidator()
        ## Parameters for one dip
        self.freqIn.setValidator(self.onlyDouble)
        self.ampIn.setValidator(self.onlyDouble)
        self.lineIn.setValidator(self.onlyDouble)
        self.lineLow.setValidator(self.onlyDouble)
        self.lineHi.setValidator(self.onlyDouble)
        self.contrastLow.setValidator(self.onlyDouble)
        self.contrastHi.setValidator(self.onlyDouble)
        self.ampLow.setValidator(self.onlyDouble)
        self.ampHi.setValidator(self.onlyDouble)
        ## Parameters for nuclear fit
        self.freqIn2.setValidator(self.onlyDouble)
        self.splitIn.setValidator(self.onlyDouble)
        self.amp1In.setValidator(self.onlyDouble)
        self.amp2In.setValidator(self.onlyDouble)
        self.line1In.setValidator(self.onlyDouble)
        self.line2In.setValidator(self.onlyDouble)

        self.testProg.setValue(0)
        # self.matplotlibwidget.canvas.mpl_connect('button_press_event', self.onclick)
        self.graphWidget.scene().sigMouseClicked.connect(self.onclick)
        self.saveButton.clicked.connect(self.save_function)
        print(self.FitOption.currentIndex())
        # load external qss style sheets, not sure if clashes with style sheet from within qtdesigner
        #sshFile = "dracula.qss"
        #with open(sshFile, "r") as fh:
        #    self.setStyleSheet(fh.read())

    def plotMapFun(self):
        self.otherwindow = OtherWindow()
        self.plot = self.otherwindow.axis.pcolormesh(self.splittings, cmap = 'seismic')
        self.otherwindow.figure.colorbar(self.plot,ax = self.otherwindow.axis)
        self.otherwindow.canvas.draw()
        self.otherwindow.show()

    def update_plots(self):
        self.graphWidget.clear()
        self.graphWidget2.clear()
        
        self.graphWidget.plot(self.freqs/1e9, myfs.mean_norm(self.fullESR_data[self.row, self.col, :]))
        self.graphWidget.plot(self.freqs/1e9, myfs.mean_norm(self.fitted_lines[self.row, self.col, :]), pen = pen)
        
        norm = (self.fullESR_data[self.row, :, :].T - np.min(self.fullESR_data[self.row, :, :],axis = 1))/(np.max(self.fullESR_data[self.row, :, :],axis =1) - np.min(self.fullESR_data[self.row, :, :],axis =1))
        split_row = pg.PlotDataItem(self.x_axis,self.splittings[self.row,:]/1e9, pen = pen)
        self.odmr_row = pg.PColorMeshItem(self.xx, self.ff/1e9, norm.T)
        single_point = pg.PlotDataItem([self.x_axis[self.col]], [self.splittings[self.row,self.col]/1e9], symbol = 'o', symbolsize = 3)
        
        v = self.graphWidget2.addPlot()
        v.addItem(self.odmr_row)
        v.addItem(split_row)
        v.addItem(single_point)


    def onclick(self, event):
        self.new_freq = self.graphWidget.plotItem.vb.mapSceneToView(event.scenePos()).x()
        self.new_freq *= 1e9
        # self.new_freq = event.scenePos().x()
        print(self.new_freq)

    def file_function(self):
        direc = r'/Users/michael/Library/CloudStorage/GoogleDrive-mh991@cam.ac.uk/Other computers/My Laptop/TEMP/2023_10_17/magscan_trace_001/20231017_magscan_trace_000.hdf5'
        path = QFileDialog.getOpenFileName(self, 'Load File', direc)
        path = path[0]
        if path.endswith('.txt'):
            print('txt file')
            split = path.split('/')
            new_path = ''
            for i in range(len(split) - 1):
                if i < len(split) - 2:
                    new_path = new_path + split[i] + "/"
                else:
                    new_path = new_path + split[i]
            self.x_axis, self.y_axis, self.X, self.Y, self.freqs, self.fullESR_data, _, _, _ = myfs.load_magscan(new_path)
        else:
            print('hdf5 file')
            self.x_axis, self.y_axis, self.X, self.Y, self.freqs, self.fullESR_data, self.meta, _, _ = myfs.load_magscan(path)

        print('File loaded successfully')
        text = self.meta
        self.x_axis = self.x_axis - self.x_axis[0]
        x_pad = np.linspace(0,self.x_axis.max(), len(self.x_axis)+1)
        f_pad = np.linspace(self.freqs.min(), self.freqs.max(), len(self.freqs)+1)
        self.ff, self.xx = np.meshgrid(f_pad,x_pad )
        #### after loading data automatically plot initial single ESR row = 0, col = 0 for preview
        self.textBrowser.setText(text)
        self.graphWidget.plot(self.freqs/1e9,myfs.mean_norm(self.fullESR_data[self.row, self.col, :]))


    def nextCol_function(self):
        if self.col <(self.fullESR_data.shape[1]-1):
            self.col = self.col + 1
            self.update_plots()
        else:
            print('End of row!')


    def prevCol_function(self):
        if self.col>0:
            self.col = self.col-1
            self.update_plots()
        else:
            print('Start of row!')


    def nextRow_function(self):
        if self.row < self.fullESR_data.shape[0]-1:
            self.row = self.row+1
            self.update_plots()
        else:
            print('Last row!')


    def prevRow_function(self):
        if self.row >0:
            self.row = self.row-1
            self.update_plots()
        else:
            print('First row!')

    def nan_function(self):
        self.splittings[self.row, self.col] = np.nan

    def fit_function(self):
        ### Do the fitting depending on active tab
        self.testProg.setMaximum(self.fullESR_data.shape[0] * self.fullESR_data.shape[1])
        self.testProg.setMinimum(0)
        if self.FitOption.currentIndex() == 0:
            ## create empty matrices to fill in data
            test = float(self.freqIn.text()) * 1e9
            ampIn = float(self.ampIn.text())
            lineIn = float(self.lineIn.text()) * 1e6
            lineLow = float(self.lineLow.text()) * 1e6
            lineHi = float(self.lineHi.text()) * 1e6
            contrastLow = float(self.contrastLow.text())
            contrastHi = float(self.contrastHi.text())
            ampLow = float(self.ampLow.text())
            ampHi = float(self.ampHi.text())
            self.splittings = np.full(self.X.shape, np.nan)
            linewidths = np.array(self.splittings)
            spectra_minima = np.array(self.splittings)
            self.fitted_lines = np.zeros(self.fullESR_data.shape)
            freq_guesses = np.array(self.splittings)
            rows, cols = self.X.shape
            bounds = ([self.freqs[0], lineLow, contrastLow, ampLow],
                      [self.freqs[-1], lineHi, contrastHi,
                       ampHi])  # Fit bounds [freq, linewidth, contrast, norm. amplitude]
            minwidth = 3e6
            minwidth = int(len(self.freqs) * minwidth / (self.freqs[-1] - self.freqs[0]))
            for row in range(rows):
                for col in range(cols):
                    tempESR = self.fullESR_data[row, col, :] / self.fullESR_data[row, col, :].mean()
                    if (row == 0) & (col == 0):
                        freq_guess = test
                    else:
                        if self.neighborSearch.isChecked() == True:
                            freq_guess = myfs.check_neighbour_pixel(self.splittings, (row, col), neighbours=1)
                        else:
                            freq_guess = self.freqs[tempESR.argmin()]
                    oppars, covmat, fitted, fail_flag = myfs.odmr_fit(self.freqs, tempESR,
                                                                  freq_guess=[freq_guess],
                                                                  dip_number=1, amp_guess=[ampIn], linewid_guess=[lineIn],
                                                                  bounds=bounds,
                                                                  height=0.02, max_nfev=1500, gtol=1e-7,
                                                                  index_parameter=(row, col))
                    if self.neighborSearch.isChecked() == True:
                        if fail_flag == 1:
                            print("Attempting minimum search.")
                            freq_guess = self.freqs[tempESR.argmin()]
                            oppars, covmat, fitted, fail_flag_redo = myfs.odmr_fit(self.freqs, tempESR,
                                                                               freq_guess=[freq_guess],
                                                                               dip_number=1, amp_guess=[ampIn],
                                                                               linewid_guess=[lineIn],
                                                                               bounds=bounds,
                                                                               height=0.02, max_nfev=5000, gtol=1e-8,
                                                                               index_parameter=(row, col))
                            fail_flag += fail_flag_redo
                        if fail_flag == 2:
                            print("Any attempt failed at ({},{}).".format(row, col))

                    self.splittings[row, col] = oppars[0]
                    linewidths[row, col] = oppars[1]
                    spectra_minima[row, col] = self.freqs[tempESR.argmin()]
                    self.fitted_lines[row, col, :] = fitted
                    freq_guesses[row, col] = freq_guess

                    self.testProg.setValue(self.testProg.value() +1)
            print("Done")
        elif self.FitOption.currentIndex() == 1:
            print(self.FitOption.currentIndex())
            self.splittings = np.full(self.X.shape, np.nan)
            self.low = np.full(self.X.shape, np.nan)
            self.high = np.zeros(self.X.shape)
            self.mid = np.zeros(self.X.shape)
            self.fitted_lines = np.zeros(self.fullESR_data.shape)
            self.freq_guesses = np.array(self.splittings)
            rows, cols = self.X.shape
            test2 = float(self.freqIn2.text()) * 1e9
            split_guess = float(self.splitIn.text()) * 1e6
            amp1In = float(self.amp1In.text())
            amp2In = float(self.amp2In.text())
            line1In = float(self.line1In.text()) * 1e6
            line2In = float(self.line2In.text()) * 1e6
            bounds1 = ([self.freqs[0]-5e6, 2.99e6, 0.018, 1e5, 0.018, 1e5, 0],[self.freqs[-1]+5e6, 3.01e6, 0.3,  1e6,  0.3, 9e5, 2] )
            for row in range(rows):
                for col in range(cols):
                    tempESR = self.fullESR_data[row, col, :] / self.fullESR_data[row, col, :].mean()
                    if (row == 0) & (col == 0):
                        freq_guess = test2
                        print(freq_guess)
                    else:
                        if self.neighborSearch.isChecked() == True:
                            freq_guess = myfs.check_neighbour_pixel(self.low, (row, col), neighbours=1)
                            # if row >= 1 & col >= 1:
                            #     freq_guess = self.freq_guesses[row, col - 1]
                            # else:
                            #     freq_guess = myfs.check_neighbour_pixel(self.low, (row, col), neighbours=1)
                        else:
                            freq_guess = self.freqs[tempESR.argmin()]
                    oppars, covmat, fitted_data, fail_flag = myfs.nucl_odmr_fit(self.freqs, tempESR,
                                                                               freq_guess=freq_guess,
                                                                               split_guess=split_guess,
                                                                               amp1_guess=amp1In,
                                                                               linewid1_guess=line1In,
                                                                               amp2_guess=amp2In,
                                                                               linewid2_guess=line2In,
                                                                               bounds=bounds1,
                                                                               maxfev=2000, gtol=1e-8,
                                                                               index_parameter=(row, col))
                    if self.neighborSearch.isChecked() == True:
                        if fail_flag == 1:
                            print("Attempting minimum search.")
                            freq_guess = self.freqs[tempESR.argmin()]
                            oppars, covmat, fitted_data, fail_flag_redo = myfs.nucl_odmr_fit(self.freqs, tempESR,
                                                                                            freq_guess=freq_guess,
                                                                                            split_guess=split_guess,
                                                                                            amp1_guess=amp1In,
                                                                                            linewid1_guess=line1In,
                                                                                            amp2_guess=amp2In,
                                                                                            linewid2_guess=line2In,
                                                                                            bounds=bounds1,
                                                                                            max_nfev=2000,
                                                                                            gtol=1e-8,
                                                                                            index_parameter=(
                                                                                            row, col))
                            fail_flag += fail_flag_redo
                        if fail_flag == 2:
                            print("Any attempt failed at ({},{}).".format(row, col))

                    self.low[row, col] = oppars[0]
                    self.high[row, col] = oppars[0] + oppars[1]
                    self.fitted_lines[row, col, :] = fitted_data
                    self.freq_guesses[row, col] = freq_guess
                    self.testProg.setValue(self.testProg.value() + 1)
            self.splittings =  (self.low+self.high)/2
            print("Done")
        
        #### plot the single fitted line in single ESR graph now showing raw and fitted data
        self.update_plots()
       
        
    def reFit_function(self):
        freq_guess = self.new_freq
        ampIn = float(self.ampIn.text())
        lineIn = float(self.lineIn.text()) * 1e6
        lineLow = float(self.lineLow.text()) * 1e6
        lineHi = float(self.lineHi.text()) * 1e6
        contrastLow = float(self.contrastLow.text())
        contrastHi = float(self.contrastHi.text())
        ampLow = float(self.ampLow.text())
        ampHi = float(self.ampHi.text())
        if self.FitOption.currentIndex() == 0:
            if self.refitRow.isChecked() == True:
                cols = np.arange(self.col, len(self.X[0, :]) - 1, 1)
                for col in cols:
                    print(col)
                    oppars_refit, _, fitted_refit, _ = myfs.odmr_fit(self.freqs,
                                                                     self.fullESR_data[self.row, col, :],
                                                                     freq_guess=[freq_guess],
                                                                     dip_number=1, amp_guess=[ampIn],
                                                                     linewid_guess=[lineIn],
                                                                     max_nfev=1500, gtol=1e-7)
                    self.splittings[self.row, col] = oppars_refit[0]
                    self.fitted_lines[self.row, col, :] = fitted_refit
                    freq_guess = oppars_refit[0]

            else:
                oppars_refit, _, fitted_refit, _ = myfs.odmr_fit(self.freqs, self.fullESR_data[self.row, self.col, :],
                                                               freq_guess=[freq_guess],
                                                               dip_number=1, amp_guess=[ampIn], linewid_guess=[lineIn],
                                                               max_nfev=1500, gtol=1e-7)
                self.splittings[self.row, self.col] = oppars_refit[0]
                self.fitted_lines[self.row, self.col, :] = fitted_refit
                freq_guess = oppars_refit[0]

        elif self.FitOption.currentIndex() == 1:
            if self.refitRow.isChecked() == True:
                cols = np.arange(self.col, len(self.X[0,:])-1,1)
                for col in cols:
                    print(col)
                    tempESR = self.fullESR_data[self.row, col, :] / self.fullESR_data[self.row, col, :].mean()
                    bounds1 = ([self.freqs[0], 2.5e6, 0.04, 5e5, 0.04, 5e5, 0], [self.freqs[-1], 3.5e6, 0.3, 5e6, 0.3, 5e6, 2])
                    oppars_refit, _, fitted_refit, _ = myfs.nucl_odmr_fit(self.freqs, tempESR,
                                                                                freq_guess=freq_guess,
                                                                                split_guess=3e6,
                                                                                amp1_guess=0.1,
                                                                                linewid1_guess=1e6,
                                                                                amp2_guess=0.1,
                                                                                linewid2_guess=1e6,
                                                                                bounds=bounds1,
                                                                                maxfev=2000, gtol=1e-8,
                                                                                index_parameter=(self.row, self.col))
                    self.low[self.row, col] = oppars_refit[0]
                    self.high[self.row, col] = oppars_refit[0] + oppars_refit[1]
                    self.splittings[self.row, col] = (self.low[self.row, col] + self.high[self.row, col]) / 2
                    self.fitted_lines[self.row, col, :] = fitted_refit
                    freq_guess = oppars_refit[0]

            else:
                tempESR = self.fullESR_data[self.row, self.col, :] / self.fullESR_data[self.row, self.col, :].mean()
                bounds1 = (
                [self.freqs[0], 2.5e6, 0.04, 5e5, 0.04, 5e5, 0], [self.freqs[-1], 3.5e6, 0.3, 5e6, 0.3, 5e6, 2])
                oppars_refit, _, fitted_refit, _ = myfs.nucl_odmr_fit(self.freqs, tempESR,
                                                                      freq_guess=freq_guess,
                                                                      split_guess=3e6,
                                                                      amp1_guess=0.1,
                                                                      linewid1_guess=1e6,
                                                                      amp2_guess=0.1,
                                                                      linewid2_guess=1e6,
                                                                      bounds=bounds1,
                                                                      maxfev=2000, gtol=1e-8,
                                                                      index_parameter=(self.row, self.col))
                self.low[self.row, self.col] = oppars_refit[0]
                self.high[self.row, self.col] = oppars_refit[0] + oppars_refit[1]
                self.splittings[self.row, self.col] = (self.low[self.row, self.col] + self.high[self.row, self.col]) / 2
                self.fitted_lines[self.row, self.col, :] = fitted_refit
                freq_guess = oppars_refit[0]

        ### Replot everything ####################################
        self.update_plots()


    def hideFunction(self):
        self.graphWidget2.clear()
        norm = (self.fullESR_data[self.row, :, :].T - np.min(self.fullESR_data[self.row, :, :],axis = 1))/(np.max(self.fullESR_data[self.row, :, :],axis =1) - np.min(self.fullESR_data[self.row, :, :],axis =1))
        self.odmr_row = pg.PColorMeshItem(self.xx, self.ff/1e9, norm.T)
        v = self.graphWidget2.addPlot()
        v.addItem(self.odmr_row)

    def unhideFunction(self):
        self.update_plots()


    def save_function(self):
        # selecting file path
        filePath, _ = QFileDialog.getSaveFileName( self, "Save ESR slice", "",
                                                  "All Files(*.*) ")
        # if file path is blank return back
        if filePath == "":
            return
        # saving ESR slice at desired path
        np.savetxt(filePath, self.splittings[:, :])

app = QApplication(sys.argv)
window = Fit()

#qtmodern.styles.dark(app)
#mw = qtmodern.windows.ModernWindow(window)
#mw.show()

app.exec_()
