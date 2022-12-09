import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import dash_bootstrap_components as dbc

from datetime import datetime
from dash import dcc, html, Input, Output, dash_table, Dash, State, ALL
from dash_bootstrap_components import Button
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

pio.templates.default = "plotly_white"

test_df = pd.read_csv("vk13var-test.csv")
pred_df = pd.read_csv("vk13var-pred.csv")
coeff_df = pd.read_csv("coeff.csv")
weight_df = pd.read_csv("weights.csv")

df_pred = test_df[test_df["sales"] != pred_df["sales"]]
categories = np.sort(df_pred["category"].unique()).tolist()
targets = np.sort(df_pred["gender"].unique()).tolist()

price_ranges = np.sort(df_pred["price_tier"].unique()).tolist()
initial_sku = df_pred[df_pred["sku_name"] == df_pred["sku_name"].unique()[0]].iloc[0, :]
end_date = (
    df_pred.loc[df_pred["sku_name"] == df_pred["sku_name"].unique()[0], "date"]
    .drop_duplicates()
    .iloc[5]
)

test_df["margin"] = (test_df["price"] - 1.0) / test_df["price"]
margins = test_df["margin"]
margins = np.sort(margins)


def calculate_canibalisation(sku, margin, comp_margin=0.17):
    real_price = (1 / (1 - margin)) * test_df.loc[
        lambda df: df["sku_name"] == sku, "cost"
    ].values[0]
    # margin_comp = round((comp_price_ratio-1)/comp_price_ratio,2)

    total_units = coeff_df.loc[lambda df: df["sku_name"] == sku, "coef"].values[0] * (
            margin - comp_margin
    )
    result = {
        "cannibalization volume": total_units,
        "canibalization margine": real_price * total_units,
        "substitute_skus": {},
    }
    g = weight_df.loc[lambda df: df["sku_name"] == sku]
    for i, data in weight_df.loc[
        lambda df: df["sku_name"] == sku, ["substitute_sku", "weight"]
    ].iterrows():
        units = total_units * data["weight"] / g["weight"].sum()
        comp_real_price = (1 / (1 - comp_margin)) * test_df.loc[
            lambda df: df["sku_name"] == data["substitute_sku"], "cost"
        ].values[0]
        result["substitute_skus"].update(
            {
                data["substitute_sku"]: {
                    "substitute units": units,
                    "sales gross margin impact": real_price * units,
                }
            }
        )
    return result


margins = test_df["margin"].drop_duplicates()
margins = np.sort(margins)

# -------------- App and small plots -----------------#

graph_layout_small = {
    "margin": {"t": 10, "b": 10, "l": 30, "r": 30},
    "height": 260,
    "width": 250,
    "legend": {"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
}

graph_layout_wide = {
    "margin": {"t": 10, "b": 10, "l": 30, "r": 30},
    "height": 240,
    "width": 480,
}

dp = float(len(margins)) / 5

ii = np.array([int(dp * n) for n in range(5)])
margins_dropbox = margins[ii].tolist()

predicted_dates_static = ['2020-06-21', '2020-06-22', '2020-06-23', '2020-06-24', '2020-06-25', '2020-06-26']

predicted_dates = (
    pred_df.loc[(pred_df["sales"].isna()), "date"].unique().tolist()
)

predicted_dates_obj = []
for d in predicted_dates:
    predicted_dates_obj.append(datetime.strptime(d, "%Y-%m-%d"))
# endregion


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    # external_stylesheets=[dbc.themes.SKETCHY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = 'Price elasticity'


# region render
def render_categorySelect():
    return [
        dbc.Label('Category'),
        dbc.Select(
            options=[{'label': a, 'value': a} for a in categories],
            value=initial_sku['category'],
            id="category_dropdown",
            persistence=True, persistence_type='memory',
        ),
    ]


mock_dashboard_data = [
    {
        'title': 'Sales',
        'value': 9.2435,
        'percent': 4.55,
        'direction': 'up'
    },
    {
        'title': 'Revenue',
        'value': 9.2435,
        'percent': 4.55,
        'direction': 'down'
    },
    {
        'title': 'Profit',
        'value': 9.2435,
        'percent': 4.55,
        'direction': 'up'
    },
    {
        'title': 'Lift',
        'value': 9.2435,
        'percent': 4.55,
        'direction': 'up'
    }
]
mock_dashboard_list = [
    {
        'title': 'Max Sales',
        'desc': 'Margins levels',
        'value': [[2.00, 0.17], [1.45, 0.33], [2.00, 0.44], [1.45, 0.52], [1.45, 0.58]]
    },
    {
        'title': 'Average Sales',
        'desc': 'Margins levels',
        'value': [[2.00, 0.17], [1.45, 0.33], [2.00, 0.44], [1.45, 0.52], [1.45, 0.58]]
    }
    ,
    {
        'title': 'Max Profit',
        'desc': 'Margins levels',
        'value': [[2.00, 0.17], [1.45, 0.33], [2.00, 0.44], [1.45, 0.52], [1.45, 0.58]]
    }
    ,
    {
        'title': 'Average Profit',
        'desc': 'Margins levels',
        'value': [[2.00, 0.17], [1.45, 0.33], [2.00, 0.44], [1.45, 0.52], [1.45, 0.58]]
    }
]
mock_diagnostic_list = [
    {
        'title': 'Sku 1',
        'desc': 'cannibalize',
        'value': [[2.00, 'units'], [1.45, 'margine']]
    },
    {
        'title': 'Sku 2',
        'desc': 'cannibalize',
        'value': [[2.00, 'units'], [1.45, 'margine']]
    },
]


def render_dashboard_blocks(el):
    return [(
        dbc.Card(
            dbc.CardBody([
                html.I('', className='bi bi-currency-dollar dashboard_block_icon'),
                html.Div(w['title'], className='dashboard_block_title'),
                html.Div(w['value'], className='dashboard_block_value'),
                html.Div([html.I('', className='bi bi-arrow-{0}'.format(w['direction'])), w['percent'],
                          html.Span('Compare to current price')],
                         className='dashboard_block_percent {0}'.format(w['direction'])),
            ]),
            className='dashboard_block m-2'
        )
    ) for w in el]


def render_dashboard_list(el, title):
    return [dbc.Card(
        [
            dbc.CardHeader(title),
            dbc.CardBody(
                html.Table(
                    html.Tbody([(
                        html.Tr([
                            *[
                                html.Td([html.Div(w['title']), html.Div(w['desc'])])
                            ],
                            *[
                                (
                                    html.Td([
                                        html.Div(i[0], className='dashboard_block_record_colored'),
                                        html.Div(i[1])
                                    ])
                                ) for i in w['value']
                            ]
                        ], className='dashboard_block_record')
                    ) for w in el]), className='dashboard_block_list'
                )
            )
        ],
        className='dashboard_block'
    )]


# endregion

main_graph = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                html.Div(
                    [dcc.Graph(id="graph_wide", className='wide-graph')]
                )
            )
        )
    )
], className='mb-3')

dashboard_data_group = dbc.Row([
    dbc.Col(children=[], className='quarter-grid col-6', id='dashboard_data_group'),
    dbc.Col(children=[], className="col-6", id='dashboard_data_list'),
], className='mb-3')

group_pf_graphs = dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                html.Div([dcc.Graph(id="graph_grid")])
            )
        )
    )
], className='mb-2')

sidebar_header = dbc.Row(
    [
        dbc.Col([html.Img(src=app.get_asset_url('logo.svg'), className='')], className='col-6'),
        dbc.Col(
            [
                dbc.Button(
                    children=html.I(className="bi bi-three-dots-vertical"),
                    className="navbar-toggler",
                    id="navbar-toggle",
                    color='link'
                ),
                dbc.Button(
                    children=html.I(className="bi bi-three-dots-vertical"),
                    className="navbar-toggler",
                    id="sidebar-toggle",
                    color='link'
                ),
            ],
            className='col-3 offset-3',
            align='center'
        )
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        html.Div(
            [
                html.Hr(),
            ],
            id="blurb",
        ),
        html.Div('Price elasticity', className='sidebar-title'),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink("Analytics", href="/", active='exact'),
                    dbc.NavLink("Diagnostics", href="/diagnostic", active="exact"),
                    dbc.Tab()
                ],
                vertical=True,
                pills=True,
            ),
            id="collapse",
        ),
    ],
    id="sidebar",
)

content = html.Div(id="page-content")

main_settings_bar = dbc.Card(
    dbc.CardBody([
        dbc.Row(
            dbc.Col(
                dbc.Button('Configuration', color='primary', id='collapse-config-button', n_clicks=0, outline=True)
            )
        ),
        dbc.Collapse(
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            render_categorySelect(),
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('Target'),
                                                        dbc.Select(
                                                            id="target_dropdown",
                                                            options=[{'label': a, 'value': a} for a in targets],
                                                            value=initial_sku["gender"],
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('Price tier'),
                                                        dbc.Select(
                                                            id="price_range_dropdown",
                                                            options=[{'label': a, 'value': a} for a in price_ranges],
                                                            value=initial_sku["price_tier"],
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('SKU'),
                                                        dbc.Select(
                                                            options=[{'label': a, 'value': a} for a in
                                                                     pred_df["sku_name"].unique().tolist()],
                                                            value=initial_sku["sku_name"],
                                                            id="sku_name_dropdown",
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=6
                                        ),
                                    ],
                                    className='mb-3'
                                ),
                                dbc.Row([
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label('Margin', id="margin_text"),
                                                    dcc.Slider(
                                                        margins_dropbox[0],
                                                        margins_dropbox[-1],
                                                        marks={
                                                            m: str(round(m, 2))
                                                            for m in margins_dropbox
                                                        },
                                                        step=None,
                                                        value=margins_dropbox[0],
                                                        id="margin_slider",
                                                        persistence=True, persistence_type='memory'
                                                    )
                                                ],
                                                style={
                                                    "width": 600,
                                                },
                                            )
                                        ]
                                    ),
                                ])
                            ]
                        )
                    )
                )
            ),
            id='collapse-config',
            is_open=False,
            className='mt-3'
        )
    ]),
    className='mb-3'
)

settings_bar = dbc.Card(
    dbc.CardBody([
        dbc.Row(
            dbc.Col(
                dbc.Button('Configuration', color='primary', id='collapse-config-button', n_clicks=0, outline=True)
            )
        ),
        dbc.Collapse(
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            render_categorySelect(),
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('Target'),
                                                        dbc.Select(
                                                            id="target_dropdown",
                                                            options=[{'label': a, 'value': a} for a in targets],
                                                            value=initial_sku["gender"],
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('Price tier'),
                                                        dbc.Select(
                                                            id="price_range_dropdown",
                                                            options=[{'label': a, 'value': a} for a in price_ranges],
                                                            value=initial_sku["price_tier"],
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('SKU'),
                                                        dbc.Select(
                                                            options=[{'label': a, 'value': a} for a in
                                                                     pred_df["sku_name"].unique().tolist()],
                                                            value=initial_sku["sku_name"],
                                                            id="sku_name_dropdown",
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('Start Date'),
                                                        dbc.Select(
                                                            options=[{'label': a, 'value': a} for a in
                                                                     predicted_dates_static],
                                                            value=initial_sku["date"],
                                                            id="start_date_dropdown",
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=2
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label('End Date'),
                                                        dbc.Select(
                                                            options=[{'label': a, 'value': a} for a in
                                                                     predicted_dates_static],
                                                            value=end_date,
                                                            id="end_date_dropdown",
                                                            persistence=True, persistence_type='memory'
                                                        )
                                                    ]
                                                )
                                            ],
                                            width=2
                                        )
                                    ],
                                    className='mb-3'
                                ),
                                dbc.Row([
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label('Margin', id="margin_text"),
                                                    dcc.Slider(
                                                        margins_dropbox[0],
                                                        margins_dropbox[-1],
                                                        marks={
                                                            m: str(round(m, 2))
                                                            for m in margins_dropbox
                                                        },
                                                        step=None,
                                                        value=margins_dropbox[0],
                                                        id="margin_slider",
                                                        persistence=True, persistence_type='memory'
                                                    )
                                                ],
                                                style={
                                                    "width": 600,
                                                },
                                            )
                                        ]
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                dbc.Label("Competitor Margin:", id="competitor_margin_text"),
                                                dcc.Slider(
                                                    margins_dropbox[0],
                                                    margins_dropbox[-1],
                                                    marks={
                                                        m: str(round(m, 2))
                                                        for m in margins_dropbox
                                                    },
                                                    step=None,
                                                    value=margins_dropbox[0],
                                                    id="competitor_margin_slider",
                                                    persistence=True, persistence_type='memory'
                                                )
                                            ],
                                            style={
                                                "width": 600,
                                            },
                                        ),
                                    )
                                ])
                            ]
                        )
                    )
                )
            ),
            id='collapse-config',
            is_open=False,
            className='mt-3'
        )
    ]),
    className='mb-3'
)

dashboard_content = dbc.Container(
    [
        main_settings_bar,
        main_graph,
        dashboard_data_group,
        # old_content
    ],
    fluid=True
)

page1 = dbc.Container(
    [
        settings_bar,
        dbc.Row([
            dbc.Col([group_pf_graphs], className='col-8'),
            dbc.Col(children=[], className="col-4", id='diagnostic_list'),
        ]),
        dbc.Row(
            dbc.Col(children=[], className='full-grid col-12', id='diagnostic_blocks')
        )
    ],
    fluid=True
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output("collapse-config", "is_open"),
    Input("collapse-config-button", "n_clicks"),
    State("collapse-config", "is_open")
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return dashboard_content
    elif pathname == "/diagnostic":
        return page1
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
)
def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""


@app.callback(
    Output("collapse", "is_open"),
    [Input("navbar-toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(Output("margin_text", "children"), Input("margin_slider", "value"))
def show_margin(price):
    return "Margin: " + str(int(round(price, 2) * 100)) + "%"


@app.callback(
    Output("competitor_margin_text", "children"),
    Input("competitor_margin_slider", "value"),
)
def show_competitor_margin(price):
    return "Competitor Margin: " + str(int(round(price, 2) * 100)) + "%"


@app.callback(
    Output("sku_name_dropdown", "options"),
    [
        Input("category_dropdown", "value"),
        Input("target_dropdown", "value"),
        Input("price_range_dropdown", "value"),
    ],
)
def update_sku_names(category, target, price_range):
    alltrue = np.ones(df_pred.shape[0]).astype(bool)
    category_idx = df_pred["category"] == category if category else alltrue
    target_idx = df_pred["gender"] == target if target else alltrue
    price_range_idx = df_pred["price_tier"] == price_range if price_range else alltrue
    idx = category_idx & target_idx & price_range_idx
    df = df_pred[idx]
    return [{'label': a, 'value': a} for a in np.sort(df["sku_name"].unique()).tolist()]
    # return np.sort(df["sku_name"].unique()).tolist()


@app.callback(
    Output(f"graph_grid", "figure"),
    [
        Input("margin_slider", "value"),
        Input("sku_name_dropdown", "value"),
        Input("start_date_dropdown", "value"),
        Input("end_date_dropdown", "value"),
    ],
)
def graphs_small(margin, sku_name, start_date, end_date):
    fig = make_subplots(specs=[[{"secondary_y": True} for j in range(3)] for k in range(2)], shared_xaxes=True,
                        shared_yaxes=True, horizontal_spacing=0.05,
                        vertical_spacing=0.1,
                        rows=2, cols=3)

    date_time_start = datetime.strptime(start_date, '%Y-%m-%d')
    date_time_end = datetime.strptime(end_date, '%Y-%m-%d')

    selected_price_range = []
    for date, str_date in zip(predicted_dates_obj, predicted_dates):
        if date >= date_time_start and date <= date_time_end:
            selected_price_range.append(str_date)

    idx_sku = df_pred["sku_name"] == sku_name
    idx_range_dates = [(df_pred["date"] == date) for date in selected_price_range]

    for i, date in enumerate(selected_price_range):
        df = df_pred[idx_range_dates[i] & idx_sku]
        showlegend = i == 0

        fig.add_trace(go.Scatter(x=df["margin"], y=df["sales"] * 1.5,
                                 mode='lines',
                                 line=dict(color="yellow", width=0.1),
                                 name='upper bound', showlegend=False),
                      secondary_y=False, row=(i // 3) + 1, col=(i % 3) + 1)

        fig.add_trace(go.Scatter(x=df["margin"], y=df["sales"] * 0.8,
                                 mode='lines',
                                 line=dict(color="yellow", width=0.1),
                                 fill='tonexty',
                                 name='sales CI', showlegend=showlegend),
                      secondary_y=False, row=(i // 3) + 1, col=(i % 3) + 1)

        fig.add_trace(
            go.Scatter(x=df["margin"], y=df["sales"], line={"color": "orange", "shape": 'spline'}, name="sales",
                       showlegend=showlegend),
            secondary_y=False, row=(i // 3) + 1, col=(i % 3) + 1)

        fig.add_trace(go.Scatter(x=df["margin"], y=df["profit"], line={"color": "rgb(49,130,189)", "shape": 'spline'},
                                 name="profit", showlegend=showlegend),
                      secondary_y=True, row=(i // 3) + 1, col=(i % 3) + 1)

        if i != 0:
            fig['layout']['xaxis{}'.format(i + 1)]['title'] = date
        else:
            fig['layout']['xaxis']['title'] = date

    for i in range(1, 7):
        fig['layout']['xaxis{}'.format(i)]['showgrid'] = False
        fig['layout']['yaxis{}'.format(i)]['showgrid'] = False
        fig['layout']['yaxis{}'.format(i + 6)]['showgrid'] = False

    fig.add_vline(margin, line={"color": "#6B6B6B",  # "shape": 'spline',
                                "width": 4})

    fig.update_layout(width=900, height=480, margin={"t": 10, "b": 10, "l": 30, "r": 30})
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=0.3
    ))

    ###############
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig['layout']['yaxis2']['showgrid'] = False
    fig.update_layout(
        # font_family="Arial",
        font_color="#464545",
        legend_title_font_color="#909090"
    )
    return fig


def evaluate_profit(row):
    profit = (row["price"] - 1.0) * row["cost"] * row["sales"]
    return profit


# region some logic
dates = np.sort(df_pred["date"].unique())
dates = dates[0:6]
idx = df_pred["date"].isin(dates)
df_pred = df_pred[idx]
df_pred["margin"] = (df_pred["price"] - 1.0) / df_pred["price"]

# Get y1, y2 maximal values over all selected days:
max_y1 = df_pred["sales"].max()
df_pred["profit"] = df_pred.apply(evaluate_profit, axis=1)
max_y2 = df_pred["profit"].max()

idx_dates = [(df_pred["date"] == date) for date in dates]

# -------------- wide plot -----------------#

x_for_graph = np.sort(test_df["date"].unique().astype(np.datetime64))
x_max, x_min = x_for_graph.max() + np.timedelta64(
    1, "D"
), x_for_graph.min() - np.timedelta64(1, "D")

known_dates = pred_df.loc[~pred_df["sales"].isna(), "date"].unique().tolist()
# predicted_dates = pred_df.loc[pred_df["sales"].isna(), "date"].unique().tolist()

y_all = test_df.loc[test_df["date"].isin(predicted_dates), "sales"]
y_min, y_max = y_all.min() - 200, y_all.max() + 200


# endregion
def closest_margin(val):
    margins_unique = test_df["margin"].unique()
    return margins_unique[np.argmin(np.abs(margins_unique - val))]


@app.callback(
    Output(f"graph_wide", "figure"),
    [Input("margin_slider", "value"), Input("sku_name_dropdown", "value")],
)
def graph_wide(margin, sku_name):
    df_sku = test_df[(test_df["sku_name"] == sku_name)]
    df_known_dates = df_sku.loc[df_sku["date"].isin(known_dates)].drop_duplicates("date")
    df_predicted_dates = df_sku.loc[df_sku["date"].isin(predicted_dates) & (df_sku["margin"] == closest_margin(margin))]

    fig = make_subplots(specs=[[{"secondary_y": True}]], shared_xaxes=False)

    fig.update_layout(legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01
    ), margin=dict(t=60, b=50, l=30, r=30)
    )

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max], secondary_y=False)
    fig.add_trace(go.Scatter(x=df_known_dates["date"],
                             y=df_known_dates["sales"],
                             line={"color": "#8E9090",  # "shape": 'spline',
                                   "width": 4},

                             marker=dict(
                                 color='#BCBCBD',  # 'LightSkyBlue',
                                 size=7,
                                 opacity=0.5,
                                 line=dict(
                                     color='#6B6B6B',  # 'MediumPurple',
                                     width=2
                                 )
                             ),
                             name="historical sales",
                             mode="lines+markers",

                             ),
                  secondary_y=False
                  )

    fig.add_trace(
        go.Scatter(x=df_known_dates["date"], y=df_known_dates["margin"],
                   line={"shape": 'hv', "color": "rgb(49,130,189)", 'dash': 'dot',  # 'dash',
                         }, name="margin"),
        secondary_y=True,
    )

    ###
    max_price = df_sku.loc[df_sku["date"].isin(predicted_dates), 'margin'].max()
    min_price = df_sku.loc[df_sku["date"].isin(predicted_dates), 'margin'].min()
    df_range_predict = df_sku.loc[df_sku["date"].isin(predicted_dates)]
    fig.add_trace(go.Scatter(x=df_predicted_dates["date"],
                             y=df_range_predict.loc[df_range_predict['margin'] == min_price, "sales"] * 1.5,
                             mode='lines',
                             line=dict(color="#B7F5FF", width=0.01, shape='spline'),
                             name='upper bound'),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=df_predicted_dates["date"],
                             y=df_range_predict.loc[df_range_predict['margin'] == max_price, "sales"] * 0.8,
                             mode='lines',
                             line=dict(color="#B7F5FF", width=0.01, shape='spline'),
                             fill='tonexty',
                             fillcolor='rgba(0,0,255,0.1)',
                             opacity=0.1,
                             name='elasticity range'),
                  secondary_y=False)
    ###

    fig.add_trace(go.Scatter(x=df_predicted_dates["date"], y=df_predicted_dates["sales"] * 1.5,
                             mode='lines',
                             line=dict(color="yellow", width=0.1, shape='spline'),
                             name='upper bound'),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=df_predicted_dates["date"], y=df_predicted_dates["sales"] * 0.8,
                             mode='lines',
                             line=dict(color="yellow", width=0.1, shape='spline'),
                             fill='tonexty',
                             opacity=0.1,
                             name='sales CI'),
                  secondary_y=False)

    fig.add_trace(go.Scatter(x=df_predicted_dates["date"],
                             y=df_predicted_dates["sales"],
                             line={"color": "#FFA05D", "shape": 'spline', "width": 4},

                             marker=dict(
                                 color='#FFA335',  # 'LightSkyBlue',
                                 size=7,
                                 opacity=0.9,
                                 line=dict(
                                     color='#FF6C35',  # 'MediumPurple',
                                     width=2
                                 )
                             ),
                             name="predicted sales",
                             mode="lines+markers"),
                  secondary_y=False)

    for trace in fig['data']:
        if (trace['name'] == 'upper bound'): trace['showlegend'] = False

    ###
    fig.update_xaxes(title_text="Date")

    ###############
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig['layout']['yaxis2']['showgrid'] = False
    fig.update_layout(
        # font_family="Arial",
        font_color="#464545",
        # title_font_family="Arial",
        # title_font_color="#909090",
        legend_title_font_color="#909090"
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Sales</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Margin</b>", secondary_y=True)

    return fig


@app.callback(
    Output("dashboard_data_group", "children"),
    [
        Input("sku_name_dropdown", "value"),
        Input("margin_slider", "value"),
        # Input("competitor_margin_slider", "value"),
    ],
    [State('dashboard_data_group', 'children')]
)
def update_dashboard_data(sku_name, margine, div_children):
    mar_box_rounded = [round(m, 2) for m in margins_dropbox]
    df_sku = df_pred[(df_pred["sku_name"] == sku_name)]
    grouped = df_sku.groupby("margin")
    aggregated = (
        grouped.agg(
            sales_max=pd.NamedAgg(column="sales", aggfunc="max"),
            sales_mean=pd.NamedAgg(column="sales", aggfunc="mean"),
            profit_max=pd.NamedAgg(column="profit", aggfunc="max"),
            profit_mean=pd.NamedAgg(column="profit", aggfunc="mean"),
        )
        .sort_values("profit_mean", ascending=False)
        .reset_index()
        .round(2)
    )
    best_margin = aggregated.loc[:, "margin"].iloc[0]
    marks = {m: str(round(m, 2)) for m in margins_dropbox}
    marks.update({best_margin: "best"})

    aggregated = aggregated.loc[
        lambda df: df["margin"].round(2).isin(mar_box_rounded)
    ].sort_values(by=["margin"])
    aggregated.index = aggregated.margin
    aggregated.drop(["margin"], inplace=True, axis=1)

    price = round(1 / (1 - margine), 1)

    #########################
    known_price = test_df[(test_df["sku_name"] == sku_name) &
                          (test_df["date"] == known_dates[-1])]['price'].values[0]

    total_volume = int(df_sku.loc[df_sku['price'].round(1) == price, 'sales'].sum())
    total_volume_known = int(df_sku.loc[df_sku['price'].round(1) == known_price, 'sales'].sum())
    volume_diff = total_volume / total_volume_known -1

    margine_profit = int(df_sku.loc[df_sku['price'].round(1) == price, 'profit'].sum())
    margine_profit_known = int(df_sku.loc[df_sku['price'].round(1) == known_price, 'profit'].sum())
    margine_diff = margine_profit / margine_profit_known - 1

    lift_profit = int(margine_profit - margine_profit_known)

    df_sku['revenue'] = df_sku['price'] * df_sku['cost'] * df_sku['sales']
    revenue = int(df_sku.loc[df_sku['price'].round(1) == price, 'revenue'].sum())
    revenue_known = int(df_sku.loc[df_sku['price'].round(1) == known_price, 'revenue'].sum())
    revenue_diff = revenue / revenue_known - 1

    data = [
        {
            'title': 'Sales',
            'value': total_volume,
            'percent': round(volume_diff, 2),
            'direction': 'up' if volume_diff >= 0 else 'down'
        },
        {
            'title': 'Revenue',
            'value': revenue,
            'percent': round(revenue_diff, 2),
            'direction': 'up' if revenue_diff >= 0 else 'down'
        },
        {
            'title': 'Profit',
            'value': margine_profit,
            'percent': round(margine_diff, 2),
            'direction': 'up' if margine_diff >= 0 else 'down'
        },
        {
            'title': 'Lift',
            'value': lift_profit,
            'percent': round(margine_diff, 2),
            'direction': 'up' if margine_diff >= 0 else 'down'
        }
    ]
    return render_dashboard_blocks(data)


@app.callback(
    Output('dashboard_data_list', "children"),
    [
        Input("sku_name_dropdown", "value"),
        Input("margin_slider", "value"),
    ],
    [State('dashboard_data_list', "children")]
)
def update_dashboard_list(sku_name, margine, div_children):
    mar_box_rounded = [round(m, 2) for m in margins_dropbox]
    df_sku = df_pred[(df_pred["sku_name"] == sku_name)]
    grouped = df_sku.groupby("margin")
    aggregated = (
        grouped.agg(
            sales_max=pd.NamedAgg(column="sales", aggfunc="max"),
            sales_mean=pd.NamedAgg(column="sales", aggfunc="mean"),
            profit_max=pd.NamedAgg(column="profit", aggfunc="max"),
            profit_mean=pd.NamedAgg(column="profit", aggfunc="mean"),
        )
        .sort_values("profit_mean", ascending=False)
        .reset_index()
        .round(2)
    )
    best_margin = aggregated.loc[:, "margin"].iloc[0]
    marks = {m: str(round(m, 2)) for m in margins_dropbox}
    marks.update({best_margin: "best"})

    aggregated = aggregated.loc[
        lambda df: df["margin"].round(2).isin(mar_box_rounded)
    ].sort_values(by=["margin"])
    aggregated.index = aggregated.margin
    aggregated.drop(["margin"], inplace=True, axis=1)

    output_of_data = aggregated.T.reset_index().rename({'index': 'Margine level'}, axis=1).to_dict('records')

    format_output = {}
    for ff in output_of_data:
        format_output[ff['Margine level']] = []
        for key, value in ff.items():
            if key != 'Margine level':
                format_output[ff['Margine level']].append([int(value), key])

    data = [
        {
            'title': 'Max Sales',
            'desc': 'Margins levels',
            'value': format_output['sales_max']
        },
        {
            'title': 'Average Sales',
            'desc': 'Margins levels',
            'value': format_output['sales_mean']
        }
        ,
        {
            'title': 'Max Profit',
            'desc': 'Margins levels',
            'value': format_output['profit_max']
        }
        ,
        {
            'title': 'Average Profit',
            'desc': 'Margins levels',
            'value': format_output['profit_mean']
        }
    ]
    return render_dashboard_list(el=data, title='Price Levels Scenarios')


@app.callback(
    Output('diagnostic_list', "children"),
    [
        Input("sku_name_dropdown", "value"),
        Input("margin_slider", "value"),
        Input("competitor_margin_slider", "value"),
    ],
    [State('diagnostic_list', "children")]
)
def update_diagnostic_list(sku_name, margine, comp_margine, div_children):
    mar_box_rounded = [round(m, 2) for m in margins_dropbox]
    df_sku = df_pred[(df_pred["sku_name"] == sku_name)]
    grouped = df_sku.groupby("margin")
    aggregated = (
        grouped.agg(
            sales_max=pd.NamedAgg(column="sales", aggfunc="max"),
            sales_mean=pd.NamedAgg(column="sales", aggfunc="mean"),
            profit_max=pd.NamedAgg(column="profit", aggfunc="max"),
            profit_mean=pd.NamedAgg(column="profit", aggfunc="mean"),
        )
        .sort_values("profit_mean", ascending=False)
        .reset_index()
        .round(2)
    )
    best_margin = aggregated.loc[:, "margin"].iloc[0]
    marks = {m: str(round(m, 2)) for m in margins_dropbox}
    marks.update({best_margin: "best"})

    aggregated = aggregated.loc[
        lambda df: df["margin"].round(2).isin(mar_box_rounded)
    ].sort_values(by=["margin"])
    aggregated.index = aggregated.margin
    aggregated.drop(["margin"], inplace=True, axis=1)

    result = calculate_canibalisation(sku_name, margine, comp_margine)
    mock_diagnostic_list = [{'title': key,
                             'desc': 'cannibalize',
                             'value': [[int(value['substitute units']), 'units'],
                                       [int(value['sales gross margin impact']), 'margine']]} for i, (key, value) in
                            enumerate(result['substitute_skus'].items()) if i < 5]

    return render_dashboard_list(el=mock_diagnostic_list, title='Cross Effects')


@app.callback(
    Output('diagnostic_blocks', 'children'),
    [
        Input("sku_name_dropdown", "value"),
        Input("margin_slider", "value"),
        Input("competitor_margin_slider", "value"),
    ],
    [State('diagnostic_blocks', 'children')]
)
def update_diagnostic_block(sku_name, margine, comp_margine, div_children):
    mar_box_rounded = [round(m, 2) for m in margins_dropbox]
    df_sku = df_pred[(df_pred["sku_name"] == sku_name)]
    grouped = df_sku.groupby("margin")
    aggregated = (
        grouped.agg(
            sales_max=pd.NamedAgg(column="sales", aggfunc="max"),
            sales_mean=pd.NamedAgg(column="sales", aggfunc="mean"),
            profit_max=pd.NamedAgg(column="profit", aggfunc="max"),
            profit_mean=pd.NamedAgg(column="profit", aggfunc="mean"),
        )
        .sort_values("profit_mean", ascending=False)
        .reset_index()
        .round(2)
    )
    best_margin = aggregated.loc[:, "margin"].iloc[0]
    marks = {m: str(round(m, 2)) for m in margins_dropbox}
    marks.update({best_margin: "best"})

    aggregated = aggregated.loc[
        lambda df: df["margin"].round(2).isin(mar_box_rounded)
    ].sort_values(by=["margin"])
    aggregated.index = aggregated.margin
    aggregated.drop(["margin"], inplace=True, axis=1)

    result = calculate_canibalisation(sku_name, margine, comp_margine)

    price = round(1 / (1 - margine), 1)

    known_price = test_df[(test_df["sku_name"] == sku_name) &
                          (test_df["date"] == known_dates[-1])]['price'].values[0]

    total_volume = int(df_sku.loc[df_sku['price'].round(1) == price, 'sales'].sum())
    total_volume_known = int(df_sku.loc[df_sku['price'].round(1) == known_price, 'sales'].sum())
    volume_diff = total_volume / total_volume_known -1

    margine_profit = int(df_sku.loc[df_sku['price'].round(1) == price, 'profit'].sum())
    margine_profit_known = int(df_sku.loc[df_sku['price'].round(1) == known_price, 'profit'].sum())
    margine_diff = margine_profit / margine_profit_known - 1

    cannibalization_volume = int(result['cannibalization volume'])
    cannibalization_margine = int(result['canibalization margine'])


    known_margin = test_df[(test_df["sku_name"] == sku_name) &
                          (test_df["date"] == known_dates[-1])]['margin'].values[0]
    result_known = calculate_canibalisation(sku_name, known_margin, comp_margine)
    cannib_vol_diff = round(1-cannibalization_volume / result_known['cannibalization volume'], 2)
    cannib_marg_diff = round(1-cannibalization_margine / result_known['canibalization margine'], 2)
    data = [
        {
            'title': 'Sales',
            'value': total_volume,
            'percent': round(volume_diff, 2),
            'direction': 'up' if volume_diff >= 0 else 'down'
        },
        {
            'title': 'Profit',
            'value': margine_profit,
            'percent': round(margine_diff, 2),
            'direction': 'up' if margine_diff >= 0 else 'down'
        },
        {
            'title': 'Cannibalization volume',
            'value': cannibalization_volume,
            'percent': cannib_vol_diff,
            'direction': 'up' if cannib_vol_diff >= 0 else 'down'
        },
        {
            'title': 'Cannibalization margine',
            'value': cannibalization_margine,
            'percent': cannib_marg_diff,
            'direction': 'up' if cannib_marg_diff >= 0 else 'down'
        }
    ]
    return render_dashboard_blocks(data)


@app.callback(
    Output('margin_slider', 'marks'),
    [
        Input("sku_name_dropdown", "value"),
        Input("margin_slider", "value"),
        # Input("competitor_margin_slider", "value"),
    ]
)
def update_best_margin_mark(sku_name, margine):
    df_sku = df_pred[(df_pred["sku_name"] == sku_name)]
    grouped = df_sku.groupby("margin")
    aggregated = (
        grouped.agg(
            sales_max=pd.NamedAgg(column="sales", aggfunc="max"),
            sales_mean=pd.NamedAgg(column="sales", aggfunc="mean"),
            profit_max=pd.NamedAgg(column="profit", aggfunc="max"),
            profit_mean=pd.NamedAgg(column="profit", aggfunc="mean"),
        )
        .sort_values("profit_mean", ascending=False)
        .reset_index()
        .round(2)
    )
    best_margin = aggregated.loc[:, "margin"].iloc[0]
    marks = {m: str(round(m, 2)) for m in margins_dropbox}
    marks.update({best_margin: "best"})
    return marks


# ------------- run app ------------------#

app.run(debug=True)  # mode='external'
