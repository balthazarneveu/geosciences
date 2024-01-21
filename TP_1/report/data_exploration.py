import pandas as pd
from pathlib import Path
import plotly.express as px
try:
    root = Path(__file__).parent.parent
except NameError:
    root = Path("")/".."

TRAIN, VALIDATION = "train", "validation"


def get_data(mode=TRAIN):
    if mode == TRAIN:
        df_logs = pd.read_parquet(root/"data/Training/logs.parquet")
        df_loc = pd.read_parquet(root/"data/Training/loc.parquet")
        df_tops = pd.read_parquet(root/"data/Training/tops.parquet")
        well_names = df_logs["wellName"].unique()
    return df_logs, df_loc, df_tops, well_names


def locate_wells(df_loc):
    fig = px.scatter_mapbox(df_loc,
                            lat="Latitude",
                            lon="Longitude",
                            zoom=8,
                            height=800,
                            width=800)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_well(df_logs, well_names):
    df = df_logs[df_logs.wellName.isin(well_names)]
    df.loc[:, 'GR'] = df['GR'].clip(0, None)
    fig = px.line(df, x='DEPTH', y='GR', color='wellName',
                  title='Gamma Ray Profile per Well')
    fig.update_layout(xaxis_title='Depth', yaxis_title='Gamma Ray Intensity')
    fig.show()


if __name__ == "__main__":
    df_logs, df_loc, df_tops, well_names = get_data()
    plot_well(df_logs, well_names[:50])
