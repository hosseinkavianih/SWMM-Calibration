import yaml
import os
import pyswmm.toolkitapi as tkai
from pyswmm.simulation import Simulation
import random
from string import Template

from pydream.parameters import FlatParam
from pydream.core import run_dream
from pydream.convergence import  Gelman_Rubin

from scipy import stats as ss
#from SEPopt import SEPopt
from statsmodels.tsa.ar_model import AutoReg

from pydream.parameters import SampledParam
from scipy.stats import uniform

import numpy as np

import pandas as pd

class Swmm:
    """
    Adapted from pystorm library: https://github.com/pystorm/pystorm
    
    SUMMARY: This class is for running a swmm simulation allowing for
    real-time control; the main method is 'run simulation'
    
    INPUTS:
        config = .yaml filepath
        action_params = if the actions are from a parameterized function, these 
                        would be the parameters
    """
    
    def __init__(self, config, action_params = None):
        self.config = yaml.load(open(config, "r"), yaml.FullLoader)
        self.sim = Simulation(self.config["model_folder"] +
                              self.config["model_name"] +".inp") # initialize simulation
        self.sim.start()

        # methods
        self.methods = {
            "depthN": self._getNodeDepth,
            "depthL": self._getLinkDepth,
            "volumeN": self._getNodeVolume,
            "volumeL": self._getLinkVolume,
            "flow": self._getLinkFlow,
            "flooding": self._getNodeFlooding,
            "inflow": self._getNodeInflow,
            "setting": self._getValvePosition,
            "total_precip": self._getRainfall
        }
        
        self.action_params = action_params
        
        # create datalog
        self.data_log = {"time":[],
                         "flow": {}, "inflow": {}, "flooding": {}, 'depthN':{}, 'setting':{}, 'total_precip': {}}
        
        if self.config["states_for_computing_objectives"] is not None:
            for entity, attribute in self.config["states_for_computing_objectives"]:
                self.data_log[attribute][entity] = []
    
        if self.config["states"] is not None:
            for entity, attribute in self.config["states"]:
                self.data_log[attribute][entity] = []
    
    def run_simulation(self):
        """
        purpose: 
            step simulation formward, applying actions; currently set up 
            to open and close valves
        output: 
            boolean indicating whether the simulation is finished
        """
        done = False
        while done == False:
            if self.action_params is not None:
                actions = self._compute_actions()
                self._take_action(actions)
            time = self.sim._model.swmm_step()
            
            # log information
            self._log_tstep()
            self.data_log['time'].append(self.sim._model.getCurrentSimulationTime())
            
            done = False if time > 0 else True # time increases till the end of the sim., then resets to 0
        self._end_and_close()
    
    def _log_tstep(self):
        for attribute in self.data_log.keys():
            if attribute != "time" and len(self.data_log[attribute]) > 0:
                for entity in self.data_log[attribute].keys():# ID
                    self.data_log[attribute][entity].append(
                        self.methods[attribute](entity))
 
        
    # ------ Valve modifications -------------------------------------------
    def _getValvePosition(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.setting.value)

    def _setValvePosition(self, ID, valve):
        return self.sim._model.setLinkSetting(ID, valve)
        
    # ------ Node Parameters  ----------------------------------------------
    def _getNodeDepth(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newDepth.value)
    
    def _getRainfall(self, ID):
        return self.sim._model.getGagePrecip(ID, tkai.RainGageResults.total_precip.value)   

    def _getNodeFlooding(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.overflow.value)

    def _getNodeLosses(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.losses.value)
    
    def _getNodeVolume(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.newVolume.value)

    def _getNodeInflow(self, ID):
        return self.sim._model.getNodeResult(ID, tkai.NodeResults.totalinflow.value)

    def _setInflow(self, ID, value):
        return self.sim._model.setNodeInflow(ID, value)

    # ------ Link modifications --------------------------------------------
    def _getLinkDepth(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newDepth.value)

    def _getLinkVolume(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newVolume.value)

    def _getLinkFlow(self, ID):
        return self.sim._model.getLinkResult(ID, tkai.LinkResults.newFlow.value)
        
    def _get_state(self):
        # create list of tuples with all state variables
        states = self.config["states_for_computing_objectives"].copy()
        states.extend(self.config["states"])
        
        state = []
        for s in states:
            entity = s[0] # name of object in swmm
            attribute = s[1] # attribute of interest (e.g. flow)
            state.append(self.methods[attribute](entity))

        state = np.asarray(state)
        
        return state
    
    def _get_lagged_states(self):
        states = self.config["lagged_state_variables"]
        lag_states = []        
        for s in states:
            entity = s[0] # name of object in swmm
            attribute = s[1] # attribute of interest (e.g. flow)
            for lag_hrs in self.config['lags_hrs']:
                ct = self.data_log['time'][-1]
                lagged_idx = 0
                dif = 0
                while dif < lag_hrs:
                    lagged_idx -= 1
                    lt = self.data_log['time'][lagged_idx]
                    dif = (ct - lt).total_seconds() / 60 / 60 # hours
                    
                lag_states.append(self.data_log[attribute][entity][lagged_idx])
                
        return lag_states
    
    def _compute_actions(self):

        actions = []
        actions.append(self.action_params[0])
        actions.append(self.action_params[1])
        
        return actions
    
    
    def export_df(self):
        for key in self.data_log.keys():
            if key == 'time':
                df = pd.DataFrame({key : self.data_log[key]})
                continue
            tmp = pd.DataFrame.from_dict(self.data_log[key])
            if len(tmp) == 0:
                continue
            new_col_names = []
            for col_name in tmp.columns:
                new_col_names.append(str(col_name) + '_' + key)
            tmp.columns = new_col_names
            
            df = df.merge(tmp, left_index=True, right_index=True)
            
        return df
        
    
    def _take_action(self, actions=None):
        if actions is not None:
            for entity, valve_position in zip(self.config["action_space"], actions):
                self._setValvePosition(entity, valve_position)
                
    def _end_and_close(self):
        """
        Terminates the simulation
        """
        self.sim._model.swmm_end()
        self.sim._model.swmm_close()
        
        
def getMetrics(sim, obs):
    eps = obs-sim
    SSE = np.sum(eps**2)
    NSE = 1. - SSE/np.sum((obs - np.mean(obs))**2)
    #LNSE = 1. - np.sum((np.log(sim+0.0001) - np.log(obs+0.0001))**2) / \
            #np.sum((np.log(sim+0.001) - np.mean(np.log(obs+0.001)))**2)
    #pBias = 100. * (np.sum(sim - obs) / np.sum(sim))
    
    return NSE #SSE, NSE, LNSE, pBias
    
list_nse = []

def likelihood(param):
    # Create a yaml template file
    print(param)
    os.chdir(r"/scratch/hk3sku/SWMM/DREAM/config9")
    with open('theta.yaml', 'r') as template_file_yaml:
        template_content_yaml = template_file_yaml.read()

    index = random.randint(1, 100000)
    
    index = index + random.randint(1, 1000000) * 2 - random.randint(1, 10000000) + 532
    
    template_yaml = Template(template_content_yaml)
    content_with_input = template_yaml.substitute(input=f'input{index}')

    # Write the content to a new file
    output_filename = f'theta{index}.yaml'
    with open(output_filename, 'w') as output_file:
        output_file.write(content_with_input)
    
    PARAM = np.array(param)
    list_nse.extend(PARAM.tolist())
     
    parameter_names = [f'Param{i+1}' for i in range(len(PARAM))]
    df = pd.DataFrame(PARAM.reshape(1,14), columns=parameter_names) 

    
    template_file = r"/scratch/hk3sku/SWMM/DREAM/template_event.inp"
    output_folder = r"/scratch/hk3sku/SWMM/DREAM/swmm_models9"
    
    with open(template_file, 'r') as f:
        template_content = f.read()

    modified_template = Template(template_content)

    row = df.iloc[0]

    param_replacements = {f'P{i+1}': str(row[f'Param{i+1}']) for i in range(len(df.columns))}

        # Substitute the placeholders in the template with the parameter values
    modified_template = modified_template.safe_substitute(param_replacements)

        # Write the modified template to a new input file
    output_file = os.path.join(output_folder, f"input{index}.inp")
    with open(output_file, 'w') as output_f:
        output_f.write(modified_template)
    #Run SWMM
    
    os.chdir(r"/scratch/hk3sku/SWMM/DREAM/")
    train_dvlpd = Swmm(config=f"/scratch/hk3sku/SWMM/DREAM/config9/theta{index}.yaml", action_params = [0.2,0.2])
    train_dvlpd.run_simulation()
    df_dvlpd = train_dvlpd.export_df()
    print(index)

    sim1 = df_dvlpd['P1_depthN'].values
    sim2 = df_dvlpd['P2_depthN'].values
    
    #sim1 = df_dvlpd['7_flow'].values + df_dvlpd['9_flow'].values
    
    os.chdir(r"/scratch/hk3sku/SWMM/DREAM/")
    obs = pd.read_csv("truth1.csv")
    #obs1 = obs['7_flow'].values + obs['9_flow'].values
    obs2 = obs['P2_depthN'].values
    obs1 = obs['P1_depthN'].values
    #obs1 = obs['flow'].values
    os.chdir(r"/scratch/hk3sku/SWMM/DREAM/swmm_models9/")
    
    file_path = f"input{index}.inp"
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print(f"File {file_path} does not exist.")
        
    file_path2 = f"input{index}.out"
    if os.path.exists(file_path2):
        os.remove(file_path2)
    else:
        print(f"File {file_path2} does not exist.")  
    
    eps1 = obs1-sim1
    #ARmodel1 = AutoReg(np.log(np.abs(eps1)+0.001),1)
    #fit1 = ARmodel1.fit()
    #resid1 = fit1.resid
    resid1 = eps1
    resid1 = np.log(np.abs(eps1)+0.01)
    params1 = ss.t.fit(resid1)
    
    logL1 = np.sum(ss.t.logpdf(resid1, params1[0], params1[1], params1[2]))

    
    
    eps2 = obs2-sim2
    #ARmodel2 = AutoReg(np.log(np.abs(eps2)+0.0001),1)
   # fit2 = ARmodel2.fit()
   # resid2 = fit2.resid
    resid2 = eps2
    
    resid2 = np.log(np.abs(eps2)+0.01)
    

    params2 = ss.t.fit(resid2)

    logL2 = np.sum(ss.t.logpdf(resid2, params2[0], params2[1], params2[2]))

    
    return logL2 + logL1

ranges = [(500, 1500), (0.225, 0.775), (0.00825, 0.01175), (0.04, 0.28),
          (0.3125, 0.6875), (2.5, 5.5), (1.75, 12.25)]



ranges2 = [(500, 1500), (500, 1500), (0.225, 0.775), (0.225, 0.775), (0.00825, 0.01175), (0.00825, 0.01175), (0.04, 0.28), (0.04, 0.28),
          (0.3125, 0.6875), (0.3125, 0.6875), (2.5, 5.5), (2.5, 5.5), (1.75, 12.25), (1.75, 12.25)]

def Latin_hypercube(minn, maxn, N):

    y = np.random.rand(N)

    x = np.zeros(N)

    idx = np.random.permutation(N)

    P = (idx+1 - y)/N

    x = minn + P * (maxn - minn)

    return x

list_array = np.zeros((15,14))

for i in range(14):
#ranges2
    list_array[:,i] = (Latin_hypercube(ranges2[i][0], ranges2[i][1], 15))
    
list_start = [list_array[0],list_array[1],list_array[2],list_array[3], list_array[4],list_array[5],list_array[6],list_array[7],list_array[8],list_array[9],list_array[10],list_array[11],list_array[12],list_array[13],list_array[14]]


num_chains = 15
num_iters = 100000

numparam = 14
# Set the priors

param1 = SampledParam(uniform, loc = ranges[0][0], scale=ranges[0][1]-ranges[0][0])
param2 = SampledParam(uniform, loc = ranges[0][0], scale=ranges[0][1]-ranges[0][0])
param3 = SampledParam(uniform, loc = ranges[1][0], scale=ranges[1][1]-ranges[1][0])
param4 = SampledParam(uniform, loc = ranges[1][0], scale=ranges[1][1]-ranges[1][0])
param5 = SampledParam(uniform, loc = ranges[2][0], scale=ranges[2][1]-ranges[2][0])
param6 = SampledParam(uniform, loc = ranges[2][0], scale=ranges[2][1]-ranges[2][0])
param7 = SampledParam(uniform, loc = ranges[3][0], scale=ranges[3][1]-ranges[3][0])
param8 = SampledParam(uniform, loc = ranges[3][0], scale=ranges[3][1]-ranges[3][0])
param9 = SampledParam(uniform, loc = ranges[4][0], scale=ranges[4][1]-ranges[4][0])
param10 = SampledParam(uniform, loc = ranges[4][0], scale=ranges[4][1]-ranges[4][0])
param11 = SampledParam(uniform, loc = ranges[5][0], scale=ranges[5][1]-ranges[5][0])
param12 = SampledParam(uniform, loc = ranges[5][0], scale=ranges[5][1]-ranges[5][0])
param13 = SampledParam(uniform, loc = ranges[6][0], scale=ranges[6][1]-ranges[6][0])
param14 = SampledParam(uniform, loc = ranges[6][0], scale=ranges[6][1]-ranges[6][0])
#param15 = SampledParam(uniform, loc = ranges[7][0], scale=ranges[7][1]-ranges[7][0])

sampled_parameter_names = [param1,param2,param3,param4,param5,param6,param7,param8,param9,param10,param11,param12,param13,param14]



sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=num_iters, nchains=num_chains, nseedchains = 200, DEpairs=5, start = list_start, start_random=False, verbose=True, parallel=False)

for chain in range(len(sampled_params)):
    np.save('DREAM_norm_sampled_params_chain_'+str(chain)+'_'+str(num_iters), sampled_params[chain])
    np.save('DREAM_norm_logps_chain_'+str(chain)+'_'+str(num_iters), log_ps[chain])

list_nse_np = np.array(list_nse, dtype=object)
np.save('list_nse_np.npy', list_nse_np)
#nseedchains = 200
#DEpairs = 
