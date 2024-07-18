import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QComboBox, QPushButton, QMessageBox, QTextEdit, QSlider
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from PIL import Image
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PySide6.QtCore import Qt

# load the dataset
file_path = r'C:\Users\Chris\Desktop\final graphs\20240401-dumping-final-updated-converted.csv'
df_new = pd.read_csv(file_path)

# replace "Unknown" municipalities with "Akrotiri British Sovereign Base Area"
df_new['municipality'] = df_new['municipality'].replace('Unknown', 'Akrotiri British Sovereign Base Area')

# classification mapping
municipality_mapping = {
    'Limassol District': 'Urban Areas',
    'Trachoni': 'Semi-Urban Areas',
    'Ypsonas Municipality': 'Semi-Rural Areas',
    'Larnaca District': 'Urban Areas',
    'Livadia': 'Semi-Rural Areas',
    'Dimos Aradippou': 'Semi-Rural Areas',
    'Oroklini': 'Semi-Urban Areas',
    'Lakatameia': 'Semi-Urban Areas',
    'Nicosia District': 'Urban Areas',
    'Strovolos': 'Urban Areas',
    'Tseri': 'Semi-Rural Areas',
    'Akrotiri British Sovereign Base Area': 'Semi-Rural Areas',
    'Lefkoşa District': 'Urban Areas'
}

# add classification column
df_new['classification'] = df_new['municipality'].map(municipality_mapping)

# treat blank fields in 'Distance to Nearest Road' and 'distance_to_residential' as zero
df_new['Distance to Nearest Road'].replace('', 0, inplace=True)
df_new['distance_to_residential'].replace('', 0, inplace=True)
df_new['Distance to Nearest Road'].fillna(0, inplace=True)
df_new['distance_to_residential'].fillna(0, inplace=True)

# add "General Results" to the classifications list
classifications = ['General Results'] + list(df_new['classification'].unique())

# list of numerical columns excluding 'gray_value', 'Distance to Nearest Road', and 'distance_to_residential'
numerical_columns = ['density', 'Elevation', 'Slope']

def save_chart(fig, title, classification):
    directory = os.path.join(os.path.expanduser('~'), 'Desktop', classification)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(os.path.join(directory, title), dpi=300, bbox_inches='tight')
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_histogram(data, column, title, xlabel, bin_width=1.25, exclude_zero=False, save=False, classification=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if exclude_zero:
        data = data[data[column] != 0]
    data = data[column].dropna()
    if data.empty:
        print(f"No valid data for {title}. Skipping plot.")
        return
    
    # calculate the number of bins based on the bin width
    max_value = data.max()
    num_bins = int(np.ceil(max_value / bin_width))
    
    # define the bins edges
    bins = np.arange(0, (num_bins + 1) * bin_width, bin_width)
    
    # plot the histogram
    n, bins, patches = ax.hist(data, bins=bins, edgecolor='black')
    norm = plt.Normalize(n.min(), n.max())

    # define custom colormap
    colors = [(1.0, 1.0, 0.6), (0.6, 0.4, 0.2)]  # light brown to light yellow
    cmap = LinearSegmentedColormap.from_list("Custom", colors)

    for count, patch in zip(n, patches):
        plt.setp(patch, 'facecolor', cmap(norm(count)))
    
    plt.title(title, y=1.05, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    # set x-ticks to match the bin edges
    plt.xticks(ticks=bins[::int(10/bin_width)])  # Adjusting the step to maintain readability

    if exclude_zero:
        plt.text(0.95, 0.95, 'Zero values excluded', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Frequency')

    # ensure x-axis starts from 0
    ax.set_xlim(left=0)
    
    if save and classification:
        save_chart(fig, f"{title}.png", classification)
    else:
        plt.show()




def create_scatter_chart(data, x_column, y_column, title, xlabel, ylabel, save=False, classification=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    categories = data[y_column].astype('category').cat.codes
    scatter = ax.scatter(data[x_column], data[y_column], alpha=0.5, edgecolors='w', c=categories, cmap='Blues')
    ax.set_title(title, y=1.05, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Landcover Type')
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1), title='Landcover Type')
    if save and classification:
        save_chart(fig, f"{title}.png", classification)
    else:
        plt.show()

def create_density_chart(data, title, save=False, classification=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    landcover_counts = data['landcover_type'].value_counts()
    landcover_counts.plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(title, y=1.05, fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Landcover Type')
    plt.tight_layout()
    if save and classification:
        save_chart(fig, f"{title}.png", classification)
    else:
        plt.show()

def create_density_vs_frequency_chart(df, title, save=False, classification=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    grouped = df.groupby(['classification', 'Distance to Nearest Road']).size().reset_index(name='frequency')
    classifications = grouped['classification'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(classifications)))
    for classification, color in zip(classifications, colors):
        sub_df = grouped[grouped['classification'] == classification]
        ax.scatter(sub_df['frequency'], sub_df['Distance to Nearest Road'], alpha=0.5, edgecolors='w', label=classification, color=color)
    ax.set_title(title, y=1.05, fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Distance to Nearest Road (m)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.tight_layout()
    if save and classification:
        save_chart(fig, f"{title}.png", classification)
    else:
        plt.show()

def create_pie_chart(data, title, save=False, classification=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    landcover_counts = data['landcover_type'].value_counts()
    landcover_percentages = (landcover_counts / landcover_counts.sum()) * 100
    custom_colors = plt.cm.tab20.colors[:len(landcover_counts)]
    
    wedges, texts, autotexts = ax.pie(
        landcover_counts, autopct='', startangle=90, colors=custom_colors,
        wedgeprops=dict(width=0.4, edgecolor='w')
    )
    
    labels = [
        f'{landcover} ({count}) - {percentage:.1f}%' 
        for landcover, count, percentage in zip(landcover_counts.index, landcover_counts, landcover_percentages)
    ]
    
    plt.legend(wedges, labels, title="Landcover Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title(title, y=1.05, fontweight='bold')

    plt.tight_layout()
    if save and classification:
        save_chart(fig, f"{title}.png", classification)
    else:
        plt.show()


def create_classification_density_road_chart(df, save=False, classification=None):
    perio_colors = {
        'Urban Areas': '#004c6c',  
        'Semi-Urban Areas': '#8aa8b2',  
        'Semi-Rural Areas': '#d95c2d',  
    }

    # plot the scatter plot with different colors for each classification
    fig, ax = plt.subplots(figsize=(10, 7))

    # define unique colors for each classification
    classifications = df['classification'].unique()

    scatter_plots = {}  # to store scatter plot handles

    for classification in classifications:
        color = perio_colors.get(classification, '#d3d3d3')  # default to gray if classification not found
        subset = df[df['classification'] == classification]
        scatter = ax.scatter(subset['density'], subset['Distance to Nearest Road'], alpha=1.0, label=classification, color=color, s=50)
        scatter_plots[classification] = scatter  # store scatter plot handle

    # add labels and title
    ax.set_title('Πυκνότητα vs Απόσταση από τον Κοντινότερο Δρόμο (Γενικά Αποτελέσματα)', y=1.05, fontweight='bold')
    ax.set_xlabel('Πυκνότητα (τετραγωνικά μέτρα)')
    ax.set_ylabel('Απόσταση από τον Κοντινότερο Δρόμο (μέτρα)')
    
    # define the correct legend order
    classifications_order = ['Urban Areas', 'Semi-Urban Areas', 'Semi-Rural Areas']

    # create legend with the correct order
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=perio_colors['Urban Areas'], markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=perio_colors['Semi-Urban Areas'], markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=perio_colors['Semi-Rural Areas'], markersize=10)]
    labels = ['Αστικές Περιοχές', 'Ημι-Αστικές Περιοχές', 'Ημι-Αγροτικές Περιοχές']
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))

    # add a grid
    ax.grid(True)

    if save and classification:
        save_chart(fig, f"ClassificationDensityRoad.png", classification)
    else:
        plt.show()

def create_landcover_distance_road_chart(df, classification=None, save=False):
    perio_colors = {
        'Urban Areas': '#004c6c',  
        'Semi-Urban Areas': '#8aa8b2',  
        'Semi-Rural Areas': '#d95c2d',  
    }

    # create a mapping of long descriptions to brief names
    landcover_mapping = {
        'Land principally occupied by agriculture with significant areas of natural vegetation': 'Agriculture with Vegetation'
    }

    # replace long descriptions with brief names
    df['landcover_type'] = df['landcover_type'].replace(landcover_mapping)

    # plot the scatter plot with different colors for each classification
    fig, ax = plt.subplots(figsize=(12, 8))

    # define unique colors for each classification
    classifications = df['classification'].unique()

    scatter_plots = {}  # to store scatter plot handles

    for classification in classifications:
        color = perio_colors.get(classification, '#d3d3d3')  # default to gray if classification not found
        subset = df[df['classification'] == classification]
        scatter = ax.scatter(subset['Distance to Nearest Road'], subset['landcover_type'], alpha=1.0, label=classification, color=color, s=50)
        scatter_plots[classification] = scatter  # store scatter plot handle

    # add labels and title
    title = 'Landcover vs Distance to Nearest Road in General Reslts'

    ax.set_title(title, y=1.05, fontweight='bold')
    ax.set_xlabel('Distance to Nearest Road (m)')
    ax.set_ylabel('Landcover Type')
    # set y-ticks to the landcover types
    landcover_types = df['landcover_type'].unique()
    ax.set_yticks(range(len(landcover_types)))
    ax.set_yticklabels(landcover_types, rotation=0, ha='right', fontsize=10)

    # manually set the legend based on the title
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=perio_colors['Αστικές Περιοχές'], markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=perio_colors['Ημι-Αστικές Περιοχές'], markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=perio_colors['Ημι-Αγροτικές Περιοχές'], markersize=10)]
    labels = ['Αστικές Περιοχές', 'Ημι-Αστικές Περιοχές', 'Ημι-Αγροτικές Περιοχές']
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))

    # add a grid
    ax.grid(True)

    if save and classification:
        save_chart(fig, f"LandcoverDistanceRoad.png", classification)
    else:
        plt.show()



def extract_general_statistics(df):
    stats = {
        'Total Records': len(df),
        'Mean Elevation': df['Elevation'].mean(),
        'Mean Slope': df['Slope'].mean(),
        'Mean Distance to Nearest Road': df['Distance to Nearest Road'].mean(),
        'Mean Distance to Residential Areas': df['distance_to_residential'].mean(),
        'Mean Density': df['density'].mean()
    }
    return stats

def extract_extreme_statistics(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    grouped = df.groupby('classification')[numeric_columns].mean()
    extreme_stats = {
        'Highest Mean Elevation': grouped['Elevation'].idxmax(),
        'Lowest Mean Elevation': grouped['Elevation'].idxmin(),
        'Highest Mean Slope': grouped['Slope'].idxmax(),
        'Lowest Mean Slope': grouped['Slope'].idxmin(),
        'Highest Mean Distance to Nearest Road': grouped['Distance to Nearest Road'].idxmax(),
        'Lowest Mean Distance to Nearest Road': grouped['Distance to Nearest Road'].idxmin(),
        'Highest Mean Distance to Residential Areas': grouped['distance_to_residential'].idxmax(),
        'Lowest Mean Distance to Residential Areas': grouped['distance_to_residential'].idxmin(),
        'Highest Mean Density': grouped['density'].idxmax(),
        'Lowest Mean Density': grouped['density'].idxmin()
    }
    return extreme_stats

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Classification Data Visualization')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # display general statistics
        self.general_stats_text = QTextEdit()
        self.general_stats_text.setReadOnly(True)
        layout.addWidget(QLabel("Classification Statistics:"))
        layout.addWidget(self.general_stats_text)
        self.display_general_statistics(df_new)

        # create the classification list widget
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(QLabel("Select Classification:"))
        self.classification_list = QListWidget()
        self.classification_list.itemSelectionChanged.connect(self.update_statistics)
        for classification in classifications:
            self.classification_list.addItem(str(classification))
        h_layout1.addWidget(self.classification_list)
        layout.addLayout(h_layout1)

        # create the chart selection widget
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Select Chart:"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            'Distance to Nearest Road (including zero)',
            'Distance to Nearest Road (excluding zero)',
            'Distance to Residential Areas (including zero)',
            'Distance to Residential Areas (excluding zero)',
            'Density',
            'Elevation',
            'Slope',
            'Landcover',
            'Landcover-Frequency',
            'Landcover-Distance to Nearest Road',
            'Classification Comparison',
            'ClassificationDensityRoad'
        ])
        h_layout2.addWidget(self.chart_combo)
        layout.addLayout(h_layout2)

        # create the button to show the chart
        self.show_chart_button = QPushButton('Show Chart')
        self.show_chart_button.clicked.connect(self.show_chart)
        layout.addWidget(self.show_chart_button)

        # create the button to save all charts
        self.save_all_charts_button = QPushButton('Save All Charts')
        self.save_all_charts_button.clicked.connect(self.save_all_charts)
        layout.addWidget(self.save_all_charts_button)

        # Create a slider for adjusting the number of bins
        self.bin_slider = QSlider(Qt.Horizontal)
        self.bin_slider.setMinimum(10)
        self.bin_slider.setMaximum(200)
        self.bin_slider.setValue(30)
        self.bin_slider.setTickInterval(10)
        self.bin_slider.setTickPosition(QSlider.TicksBelow)
        self.bin_slider.valueChanged.connect(self.update_histogram)
        layout.addWidget(QLabel("Adjust Number of Bins:"))
        layout.addWidget(self.bin_slider)

        self.bins = self.bin_slider.value()

        self.setLayout(layout)

    def display_general_statistics(self, df):
        stats = extract_general_statistics(df)
        extreme_stats = extract_extreme_statistics(df_new)
        stats_text = "\n".join([f"{key}: {value:.2f}" for key, value in stats.items()])
        extreme_stats_text = "\n".join([f"{key}: {value}" for
        key, value in extreme_stats.items()])
        self.general_stats_text.setText(f"{stats_text}\n\n{extreme_stats_text}")

    def update_statistics(self):
        selected_classification = self.classification_list.currentItem()
        if selected_classification:
            classification = selected_classification.text()
            if classification == 'General Results':
                df_classification = df_new
            else:
                df_classification = df_new[df_new['classification'] == classification]
            self.display_general_statistics(df_classification)

    def show_chart(self):
        selected_classification = self.classification_list.currentItem()
        if not selected_classification:
            QMessageBox.warning(self, 'Warning', 'No classification selected')
            return

        classification = selected_classification.text()
        chart = self.chart_combo.currentText()
        if classification == 'General Results':
            df_classification = df_new
        else:
            df_classification = df_new[df_new['classification'] == classification]

        bins = self.bin_slider.value()

        if chart == 'Distance to Nearest Road (including zero)':
            create_histogram(df_classification, 'Distance to Nearest Road', f'Distribution of Distance to Nearest Road in {classification} (in Meters)', 'Distance to Nearest Road (m)', bins=bins)
        elif chart == 'Distance to Nearest Road (excluding zero)':
            create_histogram(df_classification, 'Distance to Nearest Road', f'Distribution of Distance to Nearest Road in {classification} (in Meters)', 'Distance to Nearest Road (m)', bin_width=1.25, exclude_zero=True)
        elif chart == 'Distance to Residential Areas (including zero)':
            create_histogram(df_classification, 'distance_to_residential', f'Distribution of Distance to Residential Areas in {classification} (in Meters)', 'Distance to residential areas (m)', bins=bins)
        elif chart == 'Distance to Residential Areas (excluding zero)':
            create_histogram(df_classification, 'distance_to_residential', f'Distribution of Distance to Residential Areas in {classification} (in Meters)', 'Distance to residential areas (m)', bins=bins, exclude_zero=True)
        elif chart == 'Density':
            create_histogram(df_classification, 'density', f'Distribution of Density in {classification}', 'Density in squared meters', bins=bins)
        elif chart == 'Elevation':
            create_histogram(df_classification, 'Elevation', f'Distribution of Elevation in {classification} (in Meters)', 'Elevation (m)', bins=bins)
        elif chart == 'Slope':
            create_histogram(df_classification, 'Slope', f'Distribution of Slope in {classification} (in Degrees)', 'Slope (degrees)', bins=bins)
        elif chart == 'Landcover':
            create_pie_chart(df_classification, f'Distribution of Landcover Types in {classification}')
        elif chart == 'Landcover-Distance to Nearest Road':
            create_landcover_distance_road_chart(df_classification, classification=classification)
        elif chart == 'Landcover-Density':
            create_scatter_chart(df_classification, 'density', 'landcover_type', f'Landcover vs Density in {classification}', 'Density', 'Landcover Type')
        elif chart == 'Landcover-Frequency':
            create_density_chart(df_classification, f'Landcover Type vs Frequency in {classification}')
        elif chart == 'Density vs Frequency':
            create_density_vs_frequency_chart(df_new, 'Density vs Frequency of Landcover by Classification')
        elif chart == 'Classification Comparison':
            create_classification_pie_chart(df_new)
        elif chart == 'ClassificationDensityRoad':
            create_classification_density_road_chart(df_classification, classification=classification)






    def save_all_charts(self):
        bins = self.bin_slider.value()
        for classification in classifications:
            if classification == 'General Results':
                df_classification = df_new
            else:
                df_classification = df_new[df_new['classification'] == classification]
                create_histogram(df_classification, 'Distance to Nearest Road', f'Distribution of Distance to Nearest Road in {classification} (in Meters)', 'Distance to Nearest Road (m)', save=True, classification=classification)
                create_histogram(df_classification, 'Distance to Nearest Road', f'Distribution of Distance to Nearest Road in {classification} (in Meters)', 'Distance to Nearest Road (m)', exclude_zero=True, save=True, classification=classification)
                create_histogram(df_classification, 'distance_to_residential', f'Distribution of Distance to Residential Areas in {classification} (in Meters)', 'Distance to residential areas (m)', save=True, classification=classification)
                create_histogram(df_classification, 'distance_to_residential', f'Distribution of Distance to Residential Areas in {classification} (in Meters)', 'Distance to residential areas (m)', exclude_zero=True, save=True, classification=classification)
                create_histogram(df_classification, 'density', f'Distribution of Density in {classification}', 'Density in squared meters', save=True, classification=classification)
                create_histogram(df_classification, 'Elevation', f'Distribution of Elevation in {classification} (in Meters)', 'Elevation (m)', save=True, classification=classification)
                create_histogram(df_classification, 'Slope', f'Distribution of Slope in {classification} (in Degrees)', 'Slope (degrees)', save=True, classification=classification)
                create_pie_chart(df_classification, f'Distribution of Landcover Types in {classification}', save=True, classification=classification)
                create_scatter_chart(df_classification, 'Distance to Nearest Road', 'landcover_type', f'Landcover vs Distance to Nearest Road in {classification}', 'Distance to Nearest Road', 'Landcover Type', save=True, classification=classification)
                create_scatter_chart(df_classification, 'density', 'landcover_type', f'Landcover vs Density in {classification}', 'Density', 'Landcover Type', save=True, classification=classification)
                create_density_chart(df_classification, f'Landcover Type vs Frequency in {classification}', save=True, classification=classification)
                create_density_vs_frequency_chart(df_new, 'Density vs Frequency of Landcover by Classification', save=True, classification=classification)
                create_classification_pie_chart(df_new, save=True, classification=classification)
                create_classification_density_road_chart(df_classification, save=True, classification=classification)

    def update_histogram(self):
        self.show_chart()

app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())




