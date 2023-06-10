from ProfilingAnalyzingScript import Func, CSVPATH, VIEW_ONLY_MODE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import fpdf
from fpdf import FPDF
from os import path

# Helper to plot 3d plots
def plot_position_3d(x1, y1, z1, x2, y2, z2, label1, label2):
    """
    Generates 3D plots
    """

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x1, y1, z1, color='blue',
            label=label1)
    ax.plot(x2, y2, z2, color='red',
            label=label2)

    ax.set(xlabel = 'x (m)')
    ax.set(ylabel = 'y (m)')
    ax.set(zlabel = 'z (m)')
    ax.set_title('')
    ax.legend()

    # ax.axes.set_xlim3d(left=-.5, right=2)
    # ax.axes.set_ylim3d(bottom=-.5, top=2)
    # ax.axes.set_zlim3d(bottom=0, top=2)
    if VIEW_ONLY_MODE:
        plt.show()
    plt.savefig(path.join(CSVPATH, 'fig_3.png'),
           transparent=False,
           facecolor='white',
           bbox_inches="tight")


def plot_two_v_one(x_label, y_label, data1_label, data2_label,
                    x_data1, y_data1, y_data2, title="", scatter=True):
    """
    Generates two line plots with common x-axis
    """
    fig = plt.figure(2)

    ax = fig.add_subplot(111)
    if scatter:
        ax.scatter(x_data1, y_data1, c='r', label=data1_label)
        ax.scatter(x_data1, y_data2, c='b', label=data2_label)
    else:
        ax.plot(x_data1, y_data1, 'r', label=data1_label)
        ax.plot(x_data1, y_data2, 'b', label=data2_label)
    ax.set(xlabel=x_label)
    ax.set(ylabel=y_label)
    ax.legend()

    plt.suptitle(title)
    if VIEW_ONLY_MODE:
        plt.show()
    plt.savefig(path.join(CSVPATH,(title + ".png")),
           transparent=False,
           facecolor='white',
           bbox_inches="tight")


def plot_one_v_one(x_data, y_data, scatter=True, title="",
                   x_label="cycle no.", y_label="runtime (micro_s)"):
    """
    Generates two line plots with common x-axis
    Scatter Plot by Default
    """
    fig = plt.figure(1)

    ax = fig.add_subplot(111)

    if scatter:
        ax.scatter(x_data, y_data, c='r')
    else:
        ax.plot(x_data, y_data, 'r')

    ax.set(xlabel=x_label)
    ax.set(ylabel=y_label)

    plt.suptitle(title)
    if VIEW_ONLY_MODE:
        plt.show()
    plt.savefig(path.join(CSVPATH,(title + ".png")),
           transparent=False,
           facecolor='white',
           bbox_inches="tight")


def convert_to_microsec(dataframe):
    for column in dataframe:
        if (column != "Cycle No.") and \
            (column != Func.SENSOR_TRACKS.name) and \
            (column != Func.FUSION_TRACKS.name):

            dataframe[column] = dataframe[column].values * 1000

    return dataframe


def create_letterhead(pdf, WIDTH):
    pdf.image(path.join(CSVPATH, "pdf_header.png"), 0, 0, WIDTH)


def create_title(title, pdf):

    # Add main title
    pdf.set_font('Helvetica', 'b', 20)
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)

    # Add date of report
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128,g=128,b=128)
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')

    # Add line break
    pdf.ln(10)


def write_to_pdf(pdf, words):

    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)

    pdf.write(5, words)


class PDF(FPDF):

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')