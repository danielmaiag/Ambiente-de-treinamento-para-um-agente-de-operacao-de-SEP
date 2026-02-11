# Criação do ambiente
import functions as ft
from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np

class RTS_env(Env):
    def __init__(self):
        """INICIALIZA O AMBIENTE E GERA A PRIMEIRA OBSERVAÇÃO"""
        
        super(RTS_env, self).__init__()
                
        # Inicialização de parâmetros e variáveis
        self.inicializar_parametros()
        
        # Definição do espaço de ações
        self.definir_espaco_acoes()
        
        # Definição do espaço de observação
        self.definir_espaco_observacao()
        
        # Definição do intervalo de recompensas
        self.reward_range = (-np.inf, 0)
        
        # Gerar sequência de estados do sistema
        self.gerar_timeline()
        
        # Selecionar o estado inicial
        self.estado = self.estados[0]
               
        # Gerar observação a partir do estado inicial
        self.atualizar_vetores()
        self.PD *= self.CURVA_CARGA[self.indice_curva]
        self.gerar_observacao()

    def step(self, action):
        """APLICA UMA TRANSIÇÃO AO SISTEMA"""

        # # TRECHO PARA VERIFICAÇÃO DE GERAÇÃO EM BARRAS PQ

        # # Lista das barras que não devem ter geração
        # barras_somente_carga = [3, 4, 5, 6, 8, 9, 10, 11, 12, 17, 19, 20, 24]

        # # Verificação: se PG em qualquer dessas barras for maior que 1 MW (em módulo), dispara erro
        # for barra in barras_somente_carga:
        #     indice = barra - 1 
        #     if abs(self.PG[indice]) > 1:
        #         print(f" Erro: PG[{barra}] = {self.PG[indice]:.2f} MW, valor inválido para barra somente de carga.")
        #         import sys; sys.exit()

        # ========================= APLICA AÇÕES ==============================
        
        # Reescalona as ações normalizadas para valores reais
        delta_ger  =  action[:13] * self.max_delta_GER
        delta_corte = action[13:] * self.max_delta_corte
        
        # Corre as barras de geração e aplica alterações
        j = 0
        for i in self.GEN_BUS:
            if i != self.INDICE_SW:
                self.PG[i] += delta_ger[j]
                j += 1
                
        # Clipa após aplicar as alterações para garantir que todos os geradores 
        # NSW entrem no fluxo de potência dentro dos limites do estado
        self.PG = np.clip(self.PG, self.CGMIN_BUS, self.CGMAX_BUS)
        
        # Corre as barras com carga conectada e aplica o corte de carga
        for j, i in enumerate(self.indices_load):
            self.PD[i] += delta_corte[j]
            
        # Normaliza a potência demandada na barra entre os limites
        self.PD_ideal = self.PD_ORIG * self.CURVA_CARGA[self.indice_curva]
        self.PD = np.clip(self.PD, 0, self.PD_ideal)
        
        # Atualização do vetor de corte de cargas
        self.P_CORTE = (self.PD_ideal - self.PD)
        
        # ================== CALCULA O FEEDBACK DAS AÇÕES =====================
        
        # Atualiza os fluxos e a geração SW do estado intermediário
        self.executar_FPL()
        
        # Calcula a recompensa
        self.calcular_recompensa()
                              
        # ======================== AVANÇA NO TEMPO ============================
        
        # Avalia se a ação proposta pelo agente é significativa
        if np.max(np.abs(action)) > self.a_min:
            self.t += self.dt
            
        else:
            # Avança o tempo para o próximo evento relevante
            self.t = next((t for t in self.timeline if t > self.t), self.t)
            
        self.step_ep += 1
                
        # Salva dados para análise do desempenho do modelo
        if np.mod(self.step_count, 2_000) == 0:
            self.perdas_acc.append(np.sum(self.P_CORTE))
            self.PGSW_acc.append(self.PG[self.INDICE_SW])
            self.overflow_acc.append(np.max(self.fluxos/self.CAP))
            self.rec.append([self.custo_geracao,self.custo_corte,self.custo_transmissao,self.custo_sw])
            
        self.step_count += 1
                
        # Avaliação e aplicação de modificações seguindo a curva de carga
        if self.t >= self.T_curva[self.indice_curva]:
            self.indice_curva += 1
            self.PD *= (self.CURVA_CARGA[self.indice_curva]/self.CURVA_CARGA[self.indice_curva-1])
            
        # Avaliação e aplicação de modificações nos estados dos componentes
        if self.t >= self.transicao[self.indice_estados]:
            self.indice_estados += 1
            self.estado_old = self.estado
            self.estado = self.estados[self.indice_estados]
            
            # Verifica se houve modificação no estado dos componentes
            if np.sum(np.abs(self.estado_old - self.estado)) != 0:
                self.atualizar_vetores()
            
        # Gera uma nova observação
        self.gerar_observacao()
        
        # Verifica se o episódio terminou
        if self.observacao[0] == 1:
            self.flag = 1
        
        if (self.observacao[0] == 0 and self.flag == 1) or self.step_ep >= self.step_max:
            self.flag = 0
            self.terminated = True
        
        # Informações extras
        info = {"fluxos": self.fluxos, "corte_total": np.sum(self.P_CORTE)}
        
        return(self.observacao, self.recompensa, self.terminated, self.truncated, info)
           
    def render(self):
        pass
    
    def reset(self, seed=None, options=None):
        self.flag = 0
        self.step_ep = 0
        self.terminated = False
        self.truncated = False
        # self.P_CORTE = np.zeros(self.NPD, dtype=float)
                
        self.info = {
                'flag': self.flag,
                'step_ep': self.step_ep,
                'time': self.t
            }
                
        return(self.observacao, self.info)
    
    def inicializar_parametros(self): 
        """LEITURA DE DADOS E AJUSTES PRELIMINARES"""
        # (REVISADO)
    
        # Entrada de dados do sistema RTS
        caminho = r'/content/drive/MyDrive/Colab_Notebooks/OPF_RL/Dados/IEEERTS79_GLOBAL.dat'
        [DBAR, DCIR, STOCCIR, STOCGEN, COSTGEN] = ft.entrada_dados(caminho)
        self.t_total = int(np.floor(14_500_000/12))
    
        # Entrada de dados da curva de carga
        caminho = r'/content/drive/MyDrive/Colab_Notebooks/OPF_RL/Dados/IEEERTS_ORIGINAL.load'
        self.CURVA_CARGA_INPUT = ft.entrada_curva(caminho)
        self.CURVA_CARGA = np.tile(self.CURVA_CARGA_INPUT, int(np.ceil(self.t_total / 8736)))[:self.t_total]
        self.CURVA_CARGA = self.CURVA_CARGA/100
    
        # Reorganização do dataframe DBAR e COSTGEN
        # Ordena os números das barras em ordem crescente
        DBAR = DBAR.sort_values(by='BARRA').reset_index(drop=True)
        COSTGEN = COSTGEN.sort_values(by='BARRA').reset_index(drop=True)
    
        # ---------------------------------------------------------------------
        # EXTRAÇÃO DE DADOS DE GERADORES
    
        self.BARRA = DBAR['BARRA'].to_numpy()
        self.NBUS = len(self.BARRA)
        self.PD = DBAR['PD (MW)'].to_numpy()
        self.PD_ORIG = self.PD.copy()
        self.TIPO = DBAR['TIPO']
        self.PG = DBAR['PG (MW)'].to_numpy()
        self.PG_ORIG = self.PG.copy()
        self.AREA = DBAR['AREA'].to_numpy()
        self.INTC = DBAR['INTC'].to_numpy()
        self.INDICE_SW = self.TIPO[self.TIPO == 'SW'].index[0]
        self.BARRA_SW = self.BARRA[self.INDICE_SW]
        self.NPD = 0
        self.PD_FILT = np.zeros(17)
        self.indices_load = []
        for i in range(0, self.NBUS):
            if self.PD[i] != 0:
                self.PD_FILT[self.NPD] = self.PD[i]
                self.indices_load.append(i)
                self.NPD += 1
        self.P_CORTE = np.zeros(self.NPD, dtype=float)
        self.custo_cortebus = 30000
    
        # Criação do vetor de tensões nas barras, de acordo com as áreas
        self.V_AREA = np.where(self.AREA == 1, 138e3, 230e3)
    
        # Extração de dados estocásticos e de custo
        self.CUSTO_GEN = COSTGEN['COST'].to_numpy()
        self.CG_MAX = COSTGEN['P_MAX (MW)'].to_numpy()
        self.CG_MIN = COSTGEN['P_MIN (MW)'].to_numpy()
        self.GEN_BUS = COSTGEN['BARRA'].to_numpy()
        self.GEN_BUS -= 1
        self.N_STGEN = len(self.GEN_BUS)
        self.NU = COSTGEN['NU'].to_numpy()
        self.CG_MAX = self.CG_MAX * self.NU
        self.CG_MIN = self.CG_MIN * self.NU
        self.NU_BUS = np.zeros(self.NBUS, dtype='int')
        self.CUSTO_GER_BUS = np.zeros(self.NBUS, dtype='float')
        self.CGMAX_BUS = np.zeros(self.NBUS, dtype='float')
        self.CGMIN_BUS = np.zeros(self.NBUS, dtype='float')

        j=0
        for i in range(0, self.NBUS):
            if (self.TIPO[i] == 'PV' or self.TIPO[i] == 'SW') and i != 13:
                self.NU_BUS[i] = self.NU[j]
                self.CUSTO_GER_BUS[i] = self.CUSTO_GEN[j]
                self.CGMAX_BUS[i] = self.CG_MAX[j]
                self.CGMIN_BUS[i] = self.CG_MIN[j]
                j+=1
                
        self.CGMAX_BUS = self.CGMAX_BUS 
        self.CGMIN_BUS = self.CGMIN_BUS 
        self.CGMAX_BUS_orig = self.CGMAX_BUS.copy()
        self.CGMIN_BUS_orig = self.CGMIN_BUS.copy()
        self.NGER = np.sum(self.NU)
        self.CLASSE = COSTGEN['CL'].to_numpy()
        self.REF_CLASSE = STOCGEN['NO'].to_numpy()
        TX_FALHA_GEN = STOCGEN['F_RATE'].to_numpy()
        MTTR_GEN = STOCGEN['MTTR'].to_numpy()
    
        # Laço para associar dados de confiabilidade aos geradores
        self.TX_FALHA_GER = np.zeros(self.NGER, dtype=float)
        self.MTTR_GER = np.zeros(self.NGER, dtype=float)
    
        i=0
        for j in range (0, self.N_STGEN):
            for k in range(len(self.REF_CLASSE)):
                if self.CLASSE[j] == self.REF_CLASSE[k]:
                    checkpoint = i
                    while i < checkpoint + self.NU[j]:
                        self.TX_FALHA_GER[i] = TX_FALHA_GEN[k]
                        self.MTTR_GER[i] = MTTR_GEN[k]
                        i += 1     
        
        # Definição das capacidades máximas de geração em barras NSW
        self.CGMIN_BUS_NSW = np.delete(self.CGMIN_BUS_orig, self.INDICE_SW)
        self.CGMIN_BUS_NSW = self.CGMIN_BUS_NSW[self.CGMIN_BUS_NSW != 0] 
        self.CGMAX_BUS_NSW = np.delete(self.CGMAX_BUS_orig, self.INDICE_SW)
        self.CGMAX_BUS_NSW = self.CGMAX_BUS_NSW[self.CGMAX_BUS_NSW != 0]
        
        # Extração de dados de circuitos
        self.DE = DCIR['BDE'].to_numpy()
        self.PARA = DCIR['BPARA'].to_numpy()
        self.NCIR = DCIR['NCIR'].to_numpy()
        self.QCIR = len(self.DE)
        self.R = DCIR['RES(%)'].to_numpy()/100
        self.X = DCIR['REAT(%)'].to_numpy()/100
        self.PHI = np.deg2rad(DCIR['DEF(GRAUS)']).to_numpy()
        self.CAP = DCIR['CAP_N (MW)'].to_numpy()
        self.TX_FALHA_CIR = STOCCIR['F_RATE'].to_numpy()
        self.MTTR_CIR = STOCCIR['MTTR'].to_numpy()
                    
        # Inicialização de outras variáveis úteis
        self.N_h = 12
        self.dt = 1/12
        self.t = 0
        self.indice_curva = 1
        self.indice_estados = 1
        self.terminated = False
        self.step_max = 288
        self.a_min = 1e-3
        self.step_ep = 0
        self.step_count = 0
        self.ind_barra_isolada = []
        self.perdas_acc = []
        self.PGSW_acc = []
        self.overflow_acc = []
        self.balance_acc = []
        self.rec = []
   
    def gerar_timeline(self):
        """ Gera a sequência de estados e linha temporal que será usada na SMC-S"""
        # (REVISADO)
        
        # Geração de lista de estados e vetor com tempos de transição entre estados
        [self.estados, self.transicao, n_estados] = ft.gerar_estados(self.NGER,     self.TX_FALHA_GER, self.MTTR_GER, self.TX_FALHA_CIR, 
                                                                     self.MTTR_CIR, self.QCIR,         self.t_total)
        
        # Criação da timeline
        self.T_curva = np.arange(0, n_estados)
        self.timeline = np.sort(np.concatenate((self.T_curva, self.transicao[1:])))
        
    def executar_FPL(self):
        """ Executa o fluxo de potência linearizado para atualizar fluxos e PG """
        self.fluxos, self.PG, self.blackout = ft.fluxo_DC(self.estado, self.conex,     self.TIPO, self.GER_ativos, self.NU_BUS, 
                                                          self.NBUS,   self.INDICE_SW, self.PD,   self.PG,         self.DE, self.PARA,
                                                          self.NCIR,   self.QCIR,      self.R,    self.X)

    def gerar_observacao(self):
        """Atualiza os dados do estado do sistema."""
        # (REVISADO)
        
        self.fluxos, self.PG, self.blackout = ft.fluxo_DC(self.estado, self.conex,     self.TIPO, self.GER_ativos, self.NU_BUS, 
                                                          self.NBUS,   self.INDICE_SW, self.PD,   self.PG,         self.DE, self.PARA,
                                                          self.NCIR,   self.QCIR,      self.R,    self.X)

        self.observacao = ft.observation(self.estado,         self.GER_ativos,  self.NU,         self.CGMIN_BUS,  self.CGMAX_BUS, 
                                         self.INDICE_SW,      self.NGER,        self.NBUS,       self.TIPO,       self.conex,          
                                         self.NU_BUS,         self.PG,          self.NPD,        self.N_STGEN,    self.P_CORTE,
                                         self.CURVA_CARGA,    self.fluxos,      self.CAP,        self.PD,         self.indices_load,
                                         self.CGMIN_BUS_orig,                   self.CGMAX_BUS_orig)
        
    def definir_espaco_acoes(self):
        """Define o espaço de ações normalizado em [-1,1]."""
        # (REVISADO)
        
        # Definição das dimensões do espaço de ações
        dim_ger = self.N_STGEN - 1  # Número de estações geradoras controláveis
        dim_corte = self.NPD        # Número de barras com carga conectada
        
        # Definição dos limites físicos para as ações
        self.max_delta_GER = 20/100 * self.CGMAX_BUS_NSW
        self.max_delta_corte = self.PD_FILT

        # Criação do espaço contínuo de ações
        self.action_space = Box(low=-1,
                                high=1,
                                shape=(dim_ger + dim_corte,),
                                dtype=np.float32)
        
    def definir_espaco_observacao(self):
        """Define o espaço de observação."""
        # (REVISADO)
        
        obs_low = np.concatenate([
            np.array([0]),                                # 1) Fase do episódio
            np.zeros(1),                                  # 2) Capacidade de geração máxima relativa na barra SW
            np.zeros(1),                                  # 3) Capacidade de geração mínima relativa na barra SW
            np.zeros(13),                                 # 4) Potências geradas em barras NSW
            np.zeros(13),                                 # 5) Capacidades disponíveis em barras NSW 
            np.zeros(self.NPD),                           # 6) Cortes
            np.zeros(self.NPD),                           # 7) Cargas nominais
            0.5 * np.ones(self.QCIR),                     # 8) Sobrecarga térmica
            np.zeros(self.QCIR)                           # 9) Estado das linhas
        ])

        obs_high = np.concatenate([
            np.array([1]),                                # 1) Fase do episódio
            np.ones(1),                                   # 2) Capacidade relativa (barra SW)
            np.ones(1),                                   # 3) Geração mínima relativa (barra_SW)
            self.CGMAX_BUS_NSW,                           # 4) Potências geradas em barras não SW
            self.CGMAX_BUS_NSW,                           # 5) Capacidades de geradores disponíveis NSW
            self.PD_FILT,                                 # 6) Cortes
            self.PD_FILT,                                 # 7) Cargas nominais
            np.ones(self.QCIR),                           # 8) Sobrecarga térmica
            np.ones(self.QCIR)                            # 9) Estado das linhas
        ])
        
        # Criação do espaço de observação
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.observation_space_shape = np.shape(self.observation_space)
        
    def calcular_recompensa(self):
        """Calcula as recompensas associadas a um estado"""
        # (REVISADO)
        
        if self.blackout == False:
            # Cálculo do custo de geração
            self.custo_geracao = 1000*(np.sum(self.PG * self.CUSTO_GER_BUS))
            
            # Cálculo do custo de corte
            self.custo_corte = np.sum(np.abs(self.P_CORTE * self.custo_cortebus))
            
            # Cálculo do custo de transmissão
            self.kl = 1.5 * 9630
            self.custo_transmissao = 0
            self.sobrecarga = np.abs(self.fluxos) - self.CAP
            for i in range(0, self.QCIR):
                if self.sobrecarga[i] > 0:
                    self.custo_transmissao += (self.kl * self.sobrecarga[i])
                
            # Cálculo do custo da barra swing
            self.ksw = 1.5 * 9630
            self.g_inf_sw = self.CGMIN_BUS[self.INDICE_SW] - self.PG[self.INDICE_SW]
            self.g_sup_sw = self.PG[self.INDICE_SW] - self.CGMAX_BUS[self.INDICE_SW]
            self.custo_sw = self.ksw * np.maximum.reduce([0, self.g_inf_sw, self.g_sup_sw])
            
            # Função recompensa
            self.krec = 285 * 9630
            self.recompensa = -(self.custo_geracao + self.custo_corte + self.custo_transmissao + self.custo_sw) / self.krec
            
        elif self.blackout == True:
            # Avança o tempo para o próximo evento
            self.t = next((t for t in self.timeline if t > self.t), self.t)
            self.truncated = True
    
    def atualizar_vetores(self):
        """Atualiza informações após mudança do estado"""
        # (REVISADO)
        
        # Atualiza o vetor com informação de conexões
        self.conex = ft.score(self.estado, self.NBUS, self.QCIR, self.DE, self.PARA)
        self.GER_ativos = ft.GER_ativos(self.estado, self.NBUS, self.TIPO, self.BARRA, self.NU)
        
        # Percorre cada uma das barras do sistema
        for i in range(self.NBUS):
            # Verifica se a barra é de geração
            if (self.TIPO[i] == 'PV' or self.TIPO[i] == 'SW') and i != 13:
                # Verifica se a barra está conectada ao sistema e atualiza os limites de geração por barra
                if self.conex[i] != 0:
                    self.CGMAX_BUS[i] = (self.GER_ativos[i]/self.NU_BUS[i]) * self.CGMAX_BUS_orig[i]
                    self.CGMIN_BUS[i] = (self.GER_ativos[i]/self.NU_BUS[i]) * self.CGMIN_BUS_orig[i]
                    
                elif self.conex[i] == 0: 
                    self.CGMAX_BUS[i] = 0
                    self.CGMIN_BUS[i] = 0
            
            # Zera as cargas que não estão conectadas eletricamente à barra SW
            if i in self.indices_load and self.conex[i] == 0:
                self.PD[i] = 0
                if i not in self.ind_barra_isolada:
                    self.ind_barra_isolada.append(i)
                
            # Reestabelece a conexão das cargas após o restabelecimento das linhas
            if i in self.ind_barra_isolada and self.conex[i] == 1:
                self.PD[i] = self.PD_ORIG[i] * self.CURVA_CARGA[self.indice_curva]/100
                self.ind_barra_isolada.remove(i)
        
        # Atualiza o vetor PG de acordo com as novas capacidades do sistema
        self.PG = np.clip(self.PG, self.CGMIN_BUS, self.CGMAX_BUS)