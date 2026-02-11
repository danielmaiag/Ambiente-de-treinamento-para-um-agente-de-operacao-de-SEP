import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------
# ENTRADA DE DADOS DO SISTEMA RTS
def entrada_dados(caminho):
    with open(caminho, 'r') as file:
        conteudo = file.read()
    
    # Dividir o conteúdo nas seções DBAR e DCIR
    dbar_dados = re.findall(r'BUS DATA\s+.*?\s*ref\s+.*?\n(.*?)(?=\n09999)', conteudo, re.DOTALL)
    dcir_dados = re.findall(r'DETERMINISTIC CIRCUIT DATA\s+.*?\s*ref\s+.*?\n(.*?)(?=\n09999)', conteudo, re.DOTALL)
    stoccir_dados = re.findall(r'STOCHASTIC CIRCUIT DATA\s+.*?\s*ref\s+.*?\n(.*?)(?=\n09999)', conteudo, re.DOTALL)
    stocgen_dados = re.findall(r'CLASSES OF GENERATING STATION DATA\s+.*?\s*ref\s+.*?\n(.*?)(?=\n09999)', conteudo, re.DOTALL)
    costgen_dados = re.findall(r'GENERATING STATION DATA_\s+.*?\s*ref\s+.*?\n(.*?)(?=\n09999)', conteudo, re.DOTALL)
    
    # -------------------------------
    # PROCESSAR OS DADOS DE DBAR
    
    dbar_linhas = dbar_dados[0].strip().split('\n')
    dbar_colunas = ['BARRA', 'NOME', 'TIPO', 'PD (MW)', 'PG (MW)', 'AREA', 'INTC', 'AREA OBJ']
    DBAR = pd.DataFrame([linha.split() for linha in dbar_linhas], columns=dbar_colunas)
    
    # Converter as colunas numéricas para o tipo adequado
    numeric_columns_dbar = ['BARRA', 'PD (MW)', 'PG (MW)', 'AREA', 'INTC', 'AREA OBJ']
    for col in numeric_columns_dbar:
        DBAR[col] = pd.to_numeric(DBAR[col], errors='coerce')  # Converte para numérico, substituindo erros por NaN
    
    # -------------------------------
    # PROCESSAR OS DADOS DE DCIR
    
    dcir_linhas = dcir_dados[0].strip().split('\n')
    
    # Garantir que a primeira linha de DCIR não seja repetida como cabeçalho
    dcir_colunas = ['BDE', 'BPARA', 'NCIR', 'RES(%)', 'REAT(%)', 'CAP_N (MW)', 'CAP_E(MW)', 'DEF(GRAUS)', 'AR', 'NOME_CIRC']
    dcir_linhas = [linha for linha in dcir_linhas if linha.strip()]  # Remove linhas vazias
    DCIR = pd.DataFrame([linha.split() for linha in dcir_linhas], columns=dcir_colunas)
    
    # Converter as colunas numéricas de DCIR para o tipo adequado
    numeric_columns_dcir = ['BDE', 'BPARA', 'NCIR', 'RES(%)', 'REAT(%)', 'CAP_N (MW)', 'CAP_E(MW)', 'DEF(GRAUS)', 'AR']
    for col in numeric_columns_dcir:
        DCIR[col] = pd.to_numeric(DCIR[col], errors='coerce')  # Converte para numérico, substituindo erros por NaN
    
    # -------------------------------
    # PROCESSAR OS DADOS DE STOCCIR
    
    stoccir_linhas = stoccir_dados[0].strip().split('\n')
    stoccir_colunas = ['BDE', 'BPARA', 'NCIR', 'F_RATE', 'MTTR']
    STOCCIR = pd.DataFrame([linha.split() for linha in stoccir_linhas], columns=stoccir_colunas)
    
    # Converter as colunas numéricas para o tipo adequado
    numeric_columns_stoccir = ['BDE', 'BPARA', 'NCIR', 'F_RATE', 'MTTR']
    for col in numeric_columns_stoccir:
        STOCCIR[col] = pd.to_numeric(STOCCIR[col], errors='coerce')  # Converte para numérico, substituindo erros por NaN
    
    # -------------------------------
    # PROCESSAR OS DADOS DE STOCGEN
    
    stocgen_linhas = stocgen_dados[0].strip().split('\n')
    stocgen_colunas = ['NO', 'NOME', 'NS', 'F_RATE', 'MTTR']
    STOCGEN = pd.DataFrame([linha.split() for linha in stocgen_linhas], columns=stocgen_colunas)
    
    # Converter as colunas numéricas para o tipo adequado
    numeric_columns_stocgen = ['NO', 'NS', 'F_RATE', 'MTTR']
    for col in numeric_columns_stocgen:
        STOCGEN[col] = pd.to_numeric(STOCGEN[col], errors='coerce')  # Converte para numérico, substituindo erros por NaN
    
    # -------------------------------
    # PROCESSAR OS DADOS DE COSTGEN
    
    costgen_linhas = costgen_dados[0].strip().split('\n')
    costgen_colunas = ['STAT', 'NOME', 'BARRA', 'NU', 'CL', 'P_MIN (MW)', 'P_MAX (MW)', 'COST']
    COSTGEN = pd.DataFrame([linha.split() for linha in costgen_linhas], columns=costgen_colunas)
    
    # Converter as colunas numéricas para o tipo adequado
    numeric_columns_costgen = ['STAT', 'BARRA', 'NU', 'CL', 'P_MIN (MW)', 'P_MAX (MW)', 'COST']
    for col in numeric_columns_costgen:
        COSTGEN[col] = pd.to_numeric(COSTGEN[col], errors='coerce')  # Converte para numérico, substituindo erros por NaN
        
    return(DBAR, DCIR, STOCCIR, STOCGEN, COSTGEN)

# ----------------------------------------------------------------------------------------
# ENTRADA DA CURVA DE CARGA
def entrada_curva(caminho):
    with open(caminho, 'r') as file:
        CURVA_CARGA = np.array([float(line.strip()) for line in file if line.strip().replace('.', '', 1).isdigit()])
    
    return(CURVA_CARGA)

# ----------------------------------------------------------------------------------------
# GERAR SEQUÊNCIA DE ESTADOS DOS COMPONENTES DO SISTEMA
def gerar_estados(NGER, TX_FALHA_GER, MTTR_GER, TX_FALHA_CIR, MTTR_CIR, QCIR, T_total):
    
    # 1) Cálculo de probabilidades de os elementos estarem ativos

    # A) GERADORES:
     
    # Criação de variáveis e conversão de valores por ano para valores por hora
    Pup_GER = np.zeros(NGER, dtype=float)
    TX_REPARO_GER = np.zeros(NGER)
    TX_FALHA_GER = TX_FALHA_GER/8766

    # Cálculo da probabilidade
    for i in range(0, NGER):
        TX_REPARO_GER[i] = 1/(MTTR_GER[i])
        Pup_GER[i] = TX_REPARO_GER[i]/(TX_REPARO_GER[i] + TX_FALHA_GER[i])

    # -----------------------------------------------------------------------------
    # B) CIRCUITOS:
        
    # Cálculo das taxas de falha e de reparo dos circuitos por hora
    TX_FALHA_CIR = TX_FALHA_CIR/8766
    TX_REPARO_CIR = 1/(MTTR_CIR[:38])

    # Cálculo das probabilidades de um circuito estar em estado ativo
    Pup_CIR = np.zeros(QCIR)
    Pup_CIR[:38] = TX_REPARO_CIR/(TX_REPARO_CIR + TX_FALHA_CIR[:38])

    # Conexões internas na subestação. Não possuem taxas de falha e reparo associadas
    Pup_CIR[38:42] = 1

    # -----------------------------------------------------------------------------
    # 2) GERAÇÃO DE ESTADOS DO SISTEMA DURANTE TODO O PERÍODO DE SIMULAÇÃO

    # Definição de parâmetros iniciais
    t = int(0)

    # Cálculo do estado inicial de cada um dos componentes
    estado_GER = []
    for i in range(0, NGER):
        estado_GER.append(np.random.choice([1,0], p=[Pup_GER[i], 1 - Pup_GER[i]]))
            
    estado_CIR = []
    for i in range(0, QCIR):
        estado_CIR.append(np.random.choice([1,0], p=[Pup_CIR[i], 1 - Pup_CIR[i]]))

    estados = []
    estado_inicial = np.concatenate((estado_GER, estado_CIR))
    estados.append(estado_inicial)

    # Cálculo do tempo que cada componente passa em cada estado
    tup_GER = np.random.exponential(1/TX_FALHA_GER)
    tdn_GER = np.random.exponential(1/TX_REPARO_GER)
    tup_CIR = np.random.exponential(1/TX_FALHA_CIR[:38])
    tdn_CIR = np.random.exponential(1/TX_REPARO_CIR)

    # -----------------------------------------------------------------------------
    # Alternar os estados no tempo

    tempo_transicao_GER = np.zeros(NGER, dtype=float)
    tempo_transicao_CIR = np.zeros(QCIR-4, dtype=float)
    transicao = []
    transicao.append(0)
    
    # Cálculo do tempo de transição para os geradores a partir do estado inicial

    for i in range(0, NGER):
        if estado_GER[i] == 1:
            tempo_transicao_GER[i] = tup_GER[i]
        else:
            tempo_transicao_GER[i] = tdn_GER[i]

    # Cálculo do tempo de transição para os circuitos a partir do estado inicial

    for i in range(0, QCIR-4):
        if estado_CIR[i] == 1:
            tempo_transicao_CIR[i] = tup_CIR[i]
        else:
            tempo_transicao_CIR[i] = tdn_CIR[i]
            
    # Vetor para concatenar os dados de geração e de circuitos
    tempo_transicao = np.hstack((tempo_transicao_GER, tempo_transicao_CIR))

    while t < T_total:
        
        # Cálculo do tempo em que acontecerá a próxima comutação
        delta_t = np.min(tempo_transicao)
        
        # Identificação do índice do elemento a ser comutado
        for i in range(0, len(tempo_transicao)):
            if tempo_transicao[i] == delta_t:
                indice = i
        
        # Desconto do tempo até a próxima transição em todos os elementos
        tempo_transicao -= delta_t
        
        # Identificação da classe do elemento (gerador ou circuito)
        if indice <= 31:
            # Comutação o estado do gerador
            estado_GER[indice] = comute(estado_GER[indice])
            # Inserção do novo tempo no gerador que teve o estado comutado
            if estado_GER[indice] == 1:
                tempo_transicao[indice] = tup_GER[indice]
            else:
                tempo_transicao[indice] = tdn_GER[indice]
        
        else:
            # Ajuste do índice do circuito
            indice_CIR = indice - NGER
            # Comutação do estado do circuito
            estado_CIR[indice_CIR] = comute(estado_CIR[indice_CIR])
            # Inserção do novo tempo no gerador que teve o estado comutado
            if estado_CIR[indice_CIR] == 1:
                tempo_transicao[indice] = tup_CIR[indice_CIR]
            else:
                tempo_transicao[indice] = tdn_CIR[indice_CIR]
        
        # Armazenamento de estados e do tempo em que a comutação ocorre
        estado = np.hstack((estado_GER,estado_CIR))
        estados.append(estado)
        t += delta_t
        transicao.append(t)

    return(estados, transicao, len(estados))

# ----------------------------------------------------------------------------------------
# FLUXO DE POTÊNCIA DC
def fluxo_DC(estado, conex, TIPO, GER_ativos, NU_bus, NBUS, INDICE_SW, PD, PG, DE, PARA, NCIR, QCIR, R, X):
    
    # Separação da variável de estado dos circuitos
    estado_CIR = estado[32:]
    blackout = False
    fluxos = np.zeros(QCIR, dtype=float)
                 
    # Conversão de unidades para pu
    S_base = 100  # [MW] 
    PG_pu = PG / S_base
    PD_pu = PD / S_base
    
    # Criação da matriz de susceptância B
    B = np.zeros((NBUS, NBUS), dtype=float)
    
    for n in range(QCIR):
        if estado_CIR[n] == 1:
            k = DE[n] - 1
            m = PARA[n] - 1
            bkm = 1 / X[n]
            B[k, k] += bkm
            B[m, m] += bkm
            B[k, m] -= bkm
            B[m, k] -= bkm
    
    # Remove a linha e a coluna da barra slack
    B_bus = np.delete(B, INDICE_SW, axis=0)
    B_bus = np.delete(B_bus, INDICE_SW, axis=1)
    
    det = np.linalg.det(B_bus)
    if(det != 0):

        # Vetor de potência injetada
        P_inj = PG_pu - PD_pu
    
        # Zera injeção de barras isoladas
        for i in range(0,NBUS):
            if conex[i] == 0:
                P_inj[i] = 0
    
        # Remove elemento correspondente à slack do vetor P_inj
        P_inj_rec = np.delete(P_inj, INDICE_SW)
    
        # Resolve o sistema linear
        theta = np.linalg.solve(B_bus, P_inj_rec)
        
        # Insere ângulo da barra slack como 0
        theta = np.insert(theta, INDICE_SW, 0)
        
        for n in range(QCIR):
                if estado_CIR[n] == 1:
                    k = DE[n] - 1
                    m = PARA[n] - 1
                    fluxos[n] = (theta[k] - theta[m]) / X[n]
              
        # Cálculo da potência injetada por meio dos fluxos
        P_inj_fluxos = np.zeros(NBUS)
        for n in range(QCIR):
            if estado_CIR[n] == 1:
                de = DE[n] - 1
                para = PARA[n] - 1
                P_inj_fluxos[de] += fluxos[n]
                P_inj_fluxos[para] -= fluxos[n]
        
        # Potência gerada total (fluxo + carga)
        PG = np.round(P_inj_fluxos + PD_pu, decimals=5) * S_base
    
    else:
        blackout = True
    
    fluxos *= S_base
    
    return fluxos, PG, blackout

# ----------------------------------------------------------------------------------------
# COMUTAR ESTADO DE COMPONENTES
def comute(x):
    if x == 0:
        x = 1
    elif x == 1:
        x = 0
    else:
        print('Erro')
        
    return x

# ----------------------------------------------------------------------------------------
# GERAR INFORMAÇÕES SOBRE O ESTADO GERADO
def GER_ativos(estado, NBUS, TIPO, BARRA, NU):
    
    # Contar a quantidade de geradores ativos por barra
    c = 0
    j = 0
    GER_ativos = np.zeros(NBUS, dtype = 'int64')

    for i in range(0, NBUS):
        if (TIPO[i] == 'PV' or TIPO[i] == 'SW') and BARRA[i] != 14:
            GER_ativos[i] = np.sum(estado[c:(c+NU[j])])
            c += NU[j]
            j += 1
            
    return(GER_ativos)

# ----------------------------------------------------------------------------------------
# QUANTIDADE DE LIGAÇÕES DA BARRA COM O SISTEMA
def score(estado, NBUS, QCIR, DE, PARA):
    estado_cir = estado[32:]
    conex = np.zeros(NBUS, dtype=float)
    for i in range(0, QCIR):
        if estado_cir[i] == 1:
            conex[DE[i]-1] += 1
            conex[PARA[i]-1] += 1
    return(conex)

# ----------------------------------------------------------------------------------------
# COMPOR O ESTADO QUE SERÁ ENTREGUE AO AGENTE
def observation(estado, GER_ativos, NU, CGMIN_BUS, CGMAX_BUS, INDICE_SW, NGER, NBUS, TIPO, conex, NU_bus,
                PG, NPD, N_ST_GEN, P_corte, CURVA_CARGA, fluxos, CAP, PD, indice_ger_corte, CGMIN_BUS_orig, 
                CGMAX_BUS_orig):
    
    # 1) Fase do episódio atual
    # N° de componentes ativos atual
    if (PG[INDICE_SW] < CGMAX_BUS[INDICE_SW]) and (PG[INDICE_SW] > CGMIN_BUS[INDICE_SW]) and (np.max(fluxos/CAP) < 1):
        Ob_1 = 0
    else:
        Ob_1 = 1
        
    
    # 2) Capacidade relativa de geração máxima na barra SW   
    Ob_2 = (CGMAX_BUS[INDICE_SW])/(CGMAX_BUS_orig[INDICE_SW])


    # 3) Capacidade relativa de geração mínima na barra SW
    Ob_3 = (CGMIN_BUS[INDICE_SW])/(CGMIN_BUS_orig[INDICE_SW])

    # 4) Potência de saída das estações geradoras não SW
    Ob_4 = np.zeros(N_ST_GEN-1, dtype=float)
    j=0
    for i in range(0, NBUS):
        if TIPO[i] == 'PV' and i != 13:
            if conex[i] != 0:
                Ob_4[j] = PG[i]
                j += 1

    # 5) Capacidade dos geradores disponíveis não conectados à barra SW
    Ob_5 = np.zeros(N_ST_GEN-1, dtype=float)
    j=0
    for i in range(0, NBUS):
        if TIPO[i] == 'PV' and i != 13:
            if conex[i] != 0:
                Ob_5[j] = CGMAX_BUS[i]
                j += 1

    # 6) Saída dos geradores de corte
    Ob_6 = np.zeros(NPD, dtype='float')
    for j,i in enumerate(indice_ger_corte):
        if conex[i] != 0:
            Ob_6[j] = P_corte[j]
        elif conex[i] == 0:
            Ob_6[j] = PD[i]
          
    # 7) Cargas nominais conectadas eletricamente à barra SW
    Ob_7 = np.zeros(NPD,dtype=float)
    j=0
    for j,i in enumerate(indice_ger_corte):
        if conex[i] != 0:
            Ob_7[j] = PD[i]
            
    # 8) Sobrecarga térmica relativa de cada uma das linhas de transmissão
    rl = np.abs(fluxos)/CAP
    Ob_8 = 1/(1 + np.exp(-rl))

    # 9) Estados das linhas de transmissão
    Ob_9 = estado[32:]

    # 10) Construir o vetor estado contendo todas as partes
    obs = np.hstack((Ob_1, Ob_2, Ob_3, Ob_4, Ob_5, Ob_6, Ob_7, Ob_8, Ob_9))
    
    return(obs)

# ------------------------------------------------------------------------------------------