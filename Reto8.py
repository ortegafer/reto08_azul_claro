import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
compA = pd.read_csv('Datos/Originales/Compresores/CompA.csv')
compB = pd.read_csv('Datos/Originales/Compresores/CompB.csv')
compC = pd.read_csv('Datos/Originales/Compresores/CompC.csv')
compD = pd.read_csv('Datos/Originales/Compresores/CompD.csv')
firewall = pd.read_csv('Datos/Transformados/df_mostrar_datos.csv')
compA['Compresor'] = 'A'
compB['Compresor'] = 'B'
compC['Compresor'] = 'C'
compD['Compresor'] = 'D'
compresores_dict = {'A': compA, 'B': compB, 'C': compC, 'D': compD}
min_temp = 0
max_temp = 38
app = dash.Dash(__name__, suppress_callback_exceptions=True)
external_css = [os.path.join(os.path.dirname(__file__), 'custom.css')]
for css in external_css:
    if os.path.exists(css):
        app.css.append_css({"external_url": "assets/custom.css"})
app.layout = html.Div([
    html.H1('Monitorización Industrial - Planta de Compresores y Red', style={'textAlign': 'center', 'color': '#e2211c', 'marginBottom': '30px', 'fontWeight': 'bold', 'letterSpacing': '1px'}),
    dcc.Interval(id='intervalo-actualizacion', interval=5*1000, n_intervals=0),
    dcc.Tabs([
        dcc.Tab(label='Compresores', children=[
            html.Div([
                html.H2('Estado General de Compresores', className='card'),
                html.Div([
                    dcc.Dropdown(
                        id='filtro-compresores',
                        options=[{'label': f'Compresor {k}', 'value': k} for k in ['A', 'B', 'C', 'D']],
                        value=['A', 'B', 'C', 'D'], multi=True,className='dash-dropdown',
                        style={'width': '100%', 'marginBottom': '10px', 'display': 'inline-block'})
                ], style={'width': '60%', 'margin': '0 auto 0 auto', 'display': 'block', 'marginBottom': '0px'}),
                html.Div([
                    dcc.RangeSlider(
                        id='filtro-temp',min=min_temp,max=max_temp,
                        value=[min_temp, max_temp],marks={str(y): str(y) for y in range(min_temp, max_temp+1)},step=1,tooltip={"placement": "bottom", "always_visible": True},
                        allowCross=False,updatemode='mouseup',className='dash-rangeslider',)
                ], style={'width': '90%', 'margin': '0 auto 20px auto', 'display': 'block'}),
                html.Div(id='panel-estado', style={'display': 'flex', 'gap': '30px', 'justifyContent': 'center', 'marginTop': '20px'}),
                html.Div(id='gauge-presion-dinamico', style={'display': 'flex', 'gap': '2%', 'marginTop': '10px', 'marginBottom': '-10px'}),
                html.Div(id='gauge-temp-dinamico', style={'display': 'flex', 'gap': '2%', 'marginTop': '0px', 'marginBottom': '15px'}),
                html.Div([
                    dcc.Graph(id='grafico-tendencia-temp', className='card', style={'width': '92%', 'minWidth': '400px'}),
                ], style={'display': 'flex', 'gap': '2%', 'marginTop': '30px'}),
                html.Div([
                    dcc.Graph(id='grafico-boxplot-presion', className='card', style={'width': '92%', 'minWidth': '400px'}),
                ], style={'display': 'flex', 'gap': '2%', 'marginTop': '30px'}),
                html.Div([
                    dcc.Graph(id='grafico-hist-potencia', className='card', style={'width': '92%', 'minWidth': '400px'}),
                ], style={'display': 'flex', 'gap': '2%', 'marginTop': '30px'}),
                html.Div([
                    html.Button('Descargar CSV', id='btn-descargar-csv', n_clicks=0, style={'marginBottom': '10px', 'backgroundColor': '#e2211c', 'color': 'white', 'fontWeight': 'bold', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '6px', 'fontSize': '16px', 'cursor': 'pointer'}),
                    dcc.Download(id='descarga-csv'),
                    dash_table.DataTable(
                        id='tabla-compresores-unificada',page_size=20,page_action='native',
                        style_table={'overflowX': 'auto', 'height': '600px', 'minWidth': '250px', 'width': '100%'},
                        style_cell={'textAlign': 'center', 'fontSize': '14px', 'height': '32px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#e2211c', 'color': 'white', 'fontSize': '18px'},)
                ], style={'marginTop': '30px'}),
            ], style={'maxWidth': '1200px', 'margin': 'auto'})]),
        dcc.Tab(label='Firewall Logs', children=[
            html.Div([
                html.H2('Análisis de Firewall Logs', className='card'),
                html.Div([
                    dcc.Dropdown(
                        id='filtro-firewall-action',
                        options=[{'label': str(a), 'value': a} for a in sorted(firewall['Action'].dropna().unique())],
                        value=sorted(firewall['Action'].dropna().unique()),multi=True,placeholder='Filtrar por Action',
                        style={'width': '60%', 'margin': '0 auto 20px auto'})]),
                dcc.Graph(id='grafico-firewall-barras', className='card'),
                dcc.Graph(id='grafico-firewall-tipo-ataque', className='card'),
                html.Div([
                    html.Button('Descargar CSV', id='btn-descargar-csv-firewall', n_clicks=0, style={'marginBottom': '10px', 'backgroundColor': '#e2211c', 'color': 'white', 'fontWeight': 'bold', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '6px', 'fontSize': '16px', 'cursor': 'pointer'}),
                    dcc.Download(id='descarga-csv-firewall'),
                    dash_table.DataTable(
                        id='tabla-firewall',columns=[{"name": i, "id": i} for i in firewall.columns],
                        data=firewall.to_dict('records'),page_size=10,
                        style_table={'overflowX': 'auto'},style_cell={'textAlign': 'center'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#e2211c', 'color': 'white', 'fontSize': '15px'},)
                ], style={'marginTop': '30px'})
            ], style={'maxWidth': '1200px', 'margin': 'auto'})])])], style={'backgroundColor': '#fff', 'minHeight': '100vh'})
PALETA_DIFERENCIADA = ['#e2211c', '#FF6F61', '#FF4500', '#8B0000']
def get_df_unificado(compresores):
    dfs = [compresores_dict[k] for k in compresores]
    return pd.concat(dfs, ignore_index=True)
@app.callback(
    Output('panel-estado', 'children'),
    Input('filtro-compresores', 'value'),
    Input('intervalo-actualizacion', 'n_intervals'))
def actualizar_panel_estado(compresores, n):
    panel = []
    orden = ['A', 'B', 'C', 'D']
    for k in orden:
        if k in compresores:
            df = compresores_dict[k]
            temp = df['Temperatura'].iloc[-1]
            presion = df['Presion'].iloc[-1]
            temp_estado = 'BIEN' if temp <= df['Temperatura'].mean() + df['Temperatura'].std() else ('ALERTA' if temp <= df['Temperatura'].mean() + 2*df['Temperatura'].std() else 'FALLO')
            pres_estado = 'BIEN' if presion <= df['Presion'].mean() + df['Presion'].std() else ('ALERTA' if presion <= df['Presion'].mean() + 2*df['Presion'].std() else 'FALLO')
            temp_color = {'BIEN': '#28a745', 'ALERTA': '#ffc107', 'FALLO': '#dc3545'}[temp_estado]
            pres_color = {'BIEN': '#28a745', 'ALERTA': '#ffc107', 'FALLO': '#dc3545'}[pres_estado]
            panel.append(html.Div([
                html.H4(f'Compresor {k}'),
                html.Div([
                    html.Div('Temperatura', style={'fontWeight': 'bold'}),
                    html.Div(temp_estado, style={'background': temp_color, 'color': 'white', 'padding': '5px', 'borderRadius': '8px', 'textAlign': 'center', 'marginBottom': '5px'}),
                    html.Div('Presión', style={'fontWeight': 'bold'}),
                    html.Div(pres_estado, style={'background': pres_color, 'color': 'white', 'padding': '5px', 'borderRadius': '8px', 'textAlign': 'center'})
                ])
            ], style={'width': '180px', 'boxShadow': '2px 2px 8px #aaa', 'padding': '10px', 'margin': '10px'}))
    return panel
@app.callback(
    Output('grafico-tendencia-temp', 'figure'),
    Input('filtro-compresores', 'value'),
    Input('filtro-temp', 'value'),
    Input('intervalo-actualizacion', 'n_intervals'))
def grafico_tendencia_temp(compresores, temp_range, n):
    fig = go.Figure()
    for i, k in enumerate(compresores):
        df = compresores_dict[k]
        df_filtrado = df[(df['Temperatura'] >= temp_range[0]) & (df['Temperatura'] <= temp_range[1])]
        fig.add_trace(go.Scatter(x=df_filtrado.index, y=df_filtrado['Temperatura'], mode='lines', name=f'Temperatura {k}',
                                 line=dict(color=PALETA_DIFERENCIADA[i % len(PALETA_DIFERENCIADA)])))
    fig.update_layout(title='Tendencia de Temperatura', xaxis_title='Índice (Tiempo)', yaxis_title='Temperatura')
    return fig
@app.callback(
    Output('grafico-boxplot-presion', 'figure'),
    Input('filtro-compresores', 'value'),
    Input('filtro-temp', 'value'),
    Input('intervalo-actualizacion', 'n_intervals'))
def grafico_boxplot_presion(compresores, temp_range, n):
    df_temp = get_df_unificado(compresores)
    df_temp = df_temp[(df_temp['Temperatura'] >= temp_range[0]) & (df_temp['Temperatura'] <= temp_range[1])]
    fig = px.box(df_temp, x='Compresor', y='Presion', color='Compresor', title='Boxplot de Presión por Compresor',
                 color_discrete_sequence=PALETA_DIFERENCIADA)
    return fig
@app.callback(
    Output('grafico-hist-potencia', 'figure'),
    Input('filtro-compresores', 'value'),
    Input('filtro-temp', 'value'),
    Input('intervalo-actualizacion', 'n_intervals'))
def grafico_hist_potencia(compresores, temp_range, n):
    df_temp = get_df_unificado(compresores)
    df_temp = df_temp[(df_temp['Temperatura'] >= temp_range[0]) & (df_temp['Temperatura'] <= temp_range[1])]
    fig = px.histogram(df_temp, x='Potencia_Medida', color='Compresor', barmode='overlay', nbins=50, title='Histograma de Potencia Medida (todos los compresores)',
                       color_discrete_sequence=PALETA_DIFERENCIADA)
    fig.update_yaxes(title_text="Frecuencia")
    return fig
@app.callback(
    Output('tabla-compresores-unificada', 'data'),
    Output('tabla-compresores-unificada', 'columns'),
    Input('filtro-compresores', 'value'),
    Input('filtro-temp', 'value'),
    Input('intervalo-actualizacion', 'n_intervals'))
def actualizar_tabla_compresores_unificada(compresores, temp_range, n):
    df_temp = get_df_unificado(compresores)
    df_temp = df_temp[(df_temp['Temperatura'] >= temp_range[0]) & (df_temp['Temperatura'] <= temp_range[1])]
    df_sample = df_temp.sample(n=min(20, len(df_temp)), random_state=None) if len(df_temp) > 0 else df_temp
    cols = df_temp.columns.tolist()
    return df_sample.to_dict('records'), [{"name": col, "id": col} for col in cols]
@app.callback(
    Output('descarga-csv', 'data'),
    Input('btn-descargar-csv', 'n_clicks'),
    State('filtro-compresores', 'value'),
    State('filtro-temp', 'value'),
    prevent_initial_call=True)
def descargar_csv(n_clicks, compresores, temp_range):
    df_temp = get_df_unificado(compresores)
    df_temp = df_temp[(df_temp['Temperatura'] >= temp_range[0]) & (df_temp['Temperatura'] <= temp_range[1])]
    return dcc.send_data_frame(df_temp.to_csv, filename="datos_compresores.csv", sep=';', encoding='utf-8-sig')
@app.callback(
    Output('gauge-temp-dinamico', 'children'),
    Input('filtro-compresores', 'value'),
    Input('filtro-temp', 'value'),
    Input('intervalo-actualizacion', 'n_intervals'))
def gauge_temp_dinamico(compresores, temp_range, n):
    temp_min = compA['Temperatura'].min()
    temp_max = 38
    orden = ['A', 'B', 'C', 'D']
    children = []
    for k, comp in zip(orden, [compA, compB, compC, compD]):
        if k in compresores:
            comp_filtrado = comp[(comp['Temperatura'] >= temp_range[0]) & (comp['Temperatura'] <= temp_range[1])]
            valor = comp_filtrado['Temperatura'].iloc[-1] if not comp_filtrado.empty else None
            fig = go.Figure(go.Indicator(
                mode="gauge+number",value=valor,title={'text': f"Temperatura {k}", 'font': {'color': 'black'}},
                gauge={
                    'axis': {'range': [temp_min, temp_max]},
                    'bar': {'color': '#e2211c'},
                    'bgcolor': 'black',},))
            fig.update_layout(title=None, template='plotly_white')
        else:
            fig = go.Figure(go.Indicator(
                mode="gauge",value=None,title={'text': ''},
                gauge={
                    'axis': {'range': [temp_min, temp_max]},'bar': {'color': '#ccc'},
                    'bgcolor': '#f5f5f5',},))
            fig.update_layout(title=None, template='plotly_white')
        children.append(html.Div([
            dcc.Graph(figure=fig, style={'boxShadow': 'none', 'border': 'none', 'background': 'none', 'height': '240px'})
        ], style={'flex': 1, 'marginTop': '0px', 'paddingTop': '0px'}))
    return children
@app.callback(
    Output('gauge-presion-dinamico', 'children'),
    Input('filtro-compresores', 'value'),
    Input('filtro-temp', 'value'),
    Input('intervalo-actualizacion', 'n_intervals'))
def gauge_presion_dinamico(compresores, temp_range, n):
    pres_min = min(compA['Presion'].min(), compB['Presion'].min(), compC['Presion'].min(), compD['Presion'].min())
    pres_max = 0.8
    orden = ['A', 'B', 'C', 'D']
    children = []
    for k, comp in zip(orden, [compA, compB, compC, compD]):
        if k in compresores:
            comp_filtrado = comp[(comp['Temperatura'] >= temp_range[0]) & (comp['Temperatura'] <= temp_range[1])]
            valor = comp_filtrado['Presion'].iloc[-1] if not comp_filtrado.empty else None
            fig = go.Figure(go.Indicator(
                mode="gauge+number",value=valor,title={'text': f"Presión {k}", 'font': {'color': 'black'}},
                gauge={
                    'axis': {'range': [pres_min, pres_max]},
                    'bar': {'color': '#e2211c'},'bgcolor': 'black',},))
            fig.update_layout(title=None, template='plotly_white')
        else:
            fig = go.Figure(go.Indicator(
                mode="gauge",value=None,title={'text': ''},
                gauge={
                    'axis': {'range': [pres_min, pres_max]},'bar': {'color': '#ccc'},'bgcolor': '#f5f5f5',
                },))
            fig.update_layout(title=None, template='plotly_white')
        children.append(html.Div([
            dcc.Graph(figure=fig, style={'boxShadow': 'none', 'border': 'none', 'background': 'none', 'height': '240px'})
        ], style={'flex': 1, 'marginBottom': '0px', 'paddingBottom': '0px'}))
    return children
@app.callback(
    Output('grafico-firewall-barras', 'figure'),
    Output('grafico-firewall-tipo-ataque', 'figure'),
    Output('tabla-firewall', 'data'),
    Input('intervalo-actualizacion', 'n_intervals'),
    Input('filtro-firewall-action', 'value'))
def actualizar_graficos_firewall(n, actions):
    df = firewall.copy()
    if actions:
        df = df[df['Action'].isin(actions)]
    if df.empty or 'Action' not in df.columns:
        return go.Figure(), go.Figure(), []
    action_order = []
    for a in ['allow', 'deny', 'drop']:
        if a in [str(x).lower() for x in df['Action'].dropna().unique()] and a not in action_order:
            real_val = next(x for x in df['Action'].dropna().unique() if str(x).lower() == a)
            action_order.append(real_val)
    for a in df['Action'].dropna().unique():
        if a not in action_order:
            action_order.append(a)
    COLOR_ALLOW = '#e2211c'
    COLOR_DENY = '#FF6F61'
    COLOR_DROP = '#FF4500'
    COLOR_OTHER = '#8B0000'
    color_map = {}
    for a in action_order:
        if str(a).lower() == 'allow':
            color_map[a] = COLOR_ALLOW
        elif str(a).lower() == 'deny':
            color_map[a] = COLOR_DENY
        elif str(a).lower() == 'drop':
            color_map[a] = COLOR_DROP
        else:
            color_map[a] = COLOR_OTHER
    fig_bytes_port_action_top10_stack = go.Figure()
    if 'Source Port' in df.columns and 'Bytes' in df.columns and 'Action' in df.columns and not df.empty:
        top10_ports = df['Source Port'].value_counts().nlargest(10).index.astype(str)
        fw_top10_ports = df[df['Source Port'].astype(str).isin(top10_ports)].copy()
        fw_top10_ports['Source Port (str)'] = fw_top10_ports['Source Port'].astype(str)
        bytes_action_counts = fw_top10_ports.groupby(['Source Port (str)', 'Action'])[['Bytes']].sum().reset_index()
        ordered_ports = sorted(top10_ports, key=lambda x: int(x), reverse=True)
        fig_bytes_port_action_top10_stack = px.bar(
            bytes_action_counts,x='Source Port (str)',y='Bytes',color='Action',barmode='stack',
            category_orders={'Action': action_order, 'Source Port (str)': ordered_ports},
            color_discrete_map=color_map,title='Bytes por Top 10 Source Port')
    tipo_col = next((c for c in df.columns if c.lower() in ['accion','action','tipo','tipo_ataque','attack_type']), None)
    if tipo_col and not df.empty:
        tipo_vals = [a for a in action_order if a in list(df[tipo_col].dropna().unique())]
        pie_colors = [color_map[a] for a in tipo_vals]
        fig_tipo = px.pie(
            df,names=tipo_col,title='Distribución por Tipo de Acción/Ataque',hole=0.65,
            category_orders={tipo_col: action_order},color_discrete_sequence=pie_colors)
        fig_tipo.update_layout(annotations=[dict(text='Acción', x=0.5, y=0.5, font_size=22, showarrow=False)])
    else:
        fig_tipo = go.Figure()
    return fig_bytes_port_action_top10_stack, fig_tipo, df.to_dict('records')
@app.callback(
    Output('descarga-csv-firewall', 'data'),
    Input('btn-descargar-csv-firewall', 'n_clicks'),
    prevent_initial_call=True)
def descargar_csv_firewall(n_clicks):
    return dcc.send_data_frame(firewall.to_csv, filename="df_mostrar_datos.csv", sep=';', encoding='utf-8-sig')
if __name__ == '__main__':
    app.run(debug=True)
