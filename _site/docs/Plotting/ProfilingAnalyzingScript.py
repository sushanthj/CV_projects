import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from profiling_utils import *

from enum import Enum
FILEPATH = "/home/sush/Fusion/FE_Sensor_Fusion/ros/src/TrackerNode/scripts/csvfiles/runtime_log_milliseconds.csv"
CSVPATH = "/home/sush/Fusion/FE_Sensor_Fusion/ros/src/TrackerNode/scripts/csvfiles/report_files"

# Flag to Enable Viewing of MatplotLib and no PDF generation
VIEW_ONLY_MODE = False

# for pdf generation
import fpdf
from fpdf import FPDF
TITLE = "Profiling Report"
WIDTH = 210
HEIGHT = 297

class Func(Enum):
    PERFORM_CYCLIC_UPDATE = 0
    CYCLIC_UPDATE_LIBRARY = 1
    SET_CURRENT_EGO_VEHICLE_STATE = 2
    EXECUTE_TIME_UPDATE = 3
    EXECUTE_ASSOCIATION = 4
    EXECUTE_MEASUREMENT_UPDATE_ON_MATCHES = 5
    DECREMENT_MATCH_COUNT_ON_UNMATCHED_TRACKS = 6
    PROCESS_ORPHAN_MEASUREMENTS = 7
    EXECUTE_TRACK_MANAGEMENT = 8
    SENSOR_TRACKS = 9
    FUSION_TRACKS = 10

def define_pdf(pdf):
    # Add lettterhead and title
    create_letterhead(pdf, WIDTH)
    create_title(TITLE, pdf)

    # Add some words to PDF
    write_to_pdf(pdf, "The runtime profiling of the Fusion Core library is conducted.")
    write_to_pdf(pdf, "The hierarchy of functions being profiled is shown below.")
    pdf.ln(15)

    # Function Hierarchy
    pdf.image(path.join(CSVPATH, "funtion_hierarchy.png"), w=70)
    pdf.ln(15)
    pdf.image(path.join(CSVPATH, "Overall_Runtime_vs_Cycles.png"), w=100)

    pdf.add_page()
    create_letterhead(pdf, WIDTH)
    pdf.ln(40)
    pdf.image(path.join(CSVPATH, "Overall_Runtime_vs_Cycles.png"), w=170)

    # Generate the PDF
    pdf.output(path.join(CSVPATH, "Profiling_Report.pdf"), 'F')


def main():

    df = pd.read_csv(FILEPATH, sep=',', header=0)

    df = convert_to_microsec(df)

    plot_one_v_one(df["Cycle No."], df[Func(2).name], scatter=False, title="Overall_Runtime_vs_Cycles")
    plot_one_v_one(df[Func.SENSOR_TRACKS.name], df[Func.CYCLIC_UPDATE_LIBRARY.name],
                   title="Overall_Runtime_vs_Sensor_Tracks", x_label="No. of Sensor Tracks",
                   y_label="CyclicUpdateLibrary Runtime (micro_s)")

    # Create PDF
    pdf = PDF() # A4 (210 by 297 mm)
    # Add Page
    pdf.add_page()

    define_pdf(pdf)

if __name__ == "__main__":
    main()
