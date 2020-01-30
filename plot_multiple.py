# adapted from https://stackoverflow.com/questions/48125674/how-to-read-multiple-csv-files-store-data-and-plot-in-one-figure-using-python
import matplotlib.pyplot as plt
import numpy as np

numFiles = 16 #Number of CSV files in your directory
separator = "," #Character that separates each value inside file
fExtension = ".csv" #Extension of the file storing the data
flips = True

def MultiplePlots(xValues, allValLosses, allValAccuracies):
    'Method to plot multiple times in one figure.'

    xValues.pop(0)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("d_dim")
    ax1.set_ylabel("validation loss")
    ax1.set_ylim(0,2.5)
    ax1.set_yticks(np.arange(0,2.6,0.25))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("validation accuracy")
    ax2.set_ylim(0,1)
    ax2.set_yticks(np.arange(0,1.1,0.1))


    for i in range(len(allValLosses)):
        valLosses = allValLosses[i][1:]
        valAccuracies = allValAccuracies[i][1:]
        red_value = (i)* (1./15)
        print(red_value)
        green_value = 1 - red_value

        ax1.plot(
            list(map(int, xValues)),
            list( map(float, valLosses)),
            label="N = " + str(i+1),
            color=(red_value, green_value, 0))

        ax2.plot(
            list(map(int, xValues)),
            list( map(float, valAccuracies)),
            label="N = " + str(i+1),
            color=(red_value, green_value, 0))

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1.12, 0.5))
    plt.savefig("flips_{}.png".format(flips))

def GetXandYValues(coordinates):
    'Method to get all coordinates from all CSV files.'
    xValues = []
    valLosses = []
    valAccuracies = []
    allValLosses = []
    allValAccuracies = []
    fst = False
    for file in coordinates:
        for coordinate in file:
            if (fst == False):
                xValues.append(coordinate[0])
            valLosses.append(coordinate[3])
            valAccuracies.append(coordinate[4])
        fst = True
        allValLosses.append(valLosses)
        allValAccuracies.append(valAccuracies)
        valLosses = []
        valAccuracies = []
    return xValues, allValLosses, allValAccuracies

def GetCoordinates( n , separator , fExtension, flips):
    'Iterates through multiple CSV files and storing X values and Y values in different Lists'
    coordinates = [] #coordinates[0] = x values --- coordinates[1] = y values
    if flips:
        folder = "flips"
    else:
        folder = "no_flips"

    for i in range(n):
        coordinates.append( FillList( ReadFile("logs/subspace_training/table13slim/" + folder + "/summary/N_{}_flips_{}_d_dim_XXXXX_lr_1.0_gamma_0.3_sched_freq_10_seed_1_epochs_30_batchsize_64.csv".format(i+1, flips)), separator ) )
    return coordinates


def ReadFile(path):
    'Function to read CSV file and store file data rows in list.'
    fileCSV = open(path,"r") #Opens file
    data = fileCSV.read() #Save file data in string
    listData = data.splitlines() #Split lines so you have List of all lines in file
    fileCSV.close() #Close file
    return listData #Return list with file's rows


def FillList(myList, separator):
    'With this method you make a list containing every row from CSV file'
    valueTemp = ""
    listTemp = []
    newList = []
    for line in myList:
        for c in line:
            if c != separator:
                valueTemp += c
            else:
                listTemp.append( valueTemp )
                valueTemp = ""
        listTemp.append( valueTemp )
        newList.append(listTemp[:])
        valueTemp = ""
        del listTemp[:]
    return newList

xValues, allValLosses, allValAccuracies = GetXandYValues( GetCoordinates( numFiles, separator , fExtension, flips) )

MultiplePlots( xValues, allValLosses, allValAccuracies )
