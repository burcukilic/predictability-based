import torch
import torch.nn as nn
import torch.distributions as D
import blocks
import torch.nn.functional as F
from tqdm import tqdm
import math

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


class EffectPredictorDeepSym(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, output_dim, state_latent_dim, action_latent_dim, deterministic=False, activation="ST"):
        super().__init__()
        self.output_dim = output_dim
        self.deterministic=deterministic
        self.state_latent_dim = state_latent_dim
        self.action_latent_dim = action_latent_dim
        self.bottleneck_dim = state_latent_dim + action_latent_dim

        self.state_encoder = nn.Sequential(
            nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_latent_dim),
            
            {"ST":blocks.STLayer(), 
             "sigmoid":nn.Tanh(), }
        )

        self.action_encoder = nn.Sequential(
            nn.BatchNorm1d(action_dim),
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_latent_dim),

            {"ST":blocks.STLayer(),
                "sigmoid":nn.Tanh(), 
                "gumbel":blocks.GumbelSigmoidLayer(deterministic=deterministic)}[activation],
        )

        self.decoder_base = nn.Sequential(
            #nn.LayerNorm(self.bottleneck_dim),
            #nn.BatchNorm1d(self.bottleneck_dim),
            blocks.MLP([self.bottleneck_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim], hiddrop=0.1),
            nn.ReLU(),
        )

        self.decoder_mean = blocks.MLP([hidden_dim, output_dim])
        self.decoder_logvar = blocks.MLP([hidden_dim, output_dim])

        for module in list(self.state_encoder.children())[:-1]:
            module.apply(self.xavier_init)

        for module in list(self.action_encoder.children())[:-1]:
            module.apply(self.xavier_init)
        
    def xavier_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_action(self, action):
        return self.action_encoder(action)
    
    def encode_state(self, state):
        return self.state_encoder(state)
    
    def decode(self, state_emb, action_emb):
        latent_symbols = self.decoder_base(torch.cat((state_emb, action_emb), dim=1))
        effect = self.decoder(latent_symbols)
        
        mean, log_var = effect.split(self.output_dim, dim=1)

        mean = self.decoder_mean(latent_symbols)
        log_var = self.decoder_logvar(latent_symbols)

        return mean, log_var

    def forward(self, state, action):
        state_emb = self.encode_state(state)  
        
        action_emb = self.encode_action(action)
        
        mean, log_var = self.decode(state_emb, action_emb)
        log_var = torch.clamp(log_var, -10, 10)

        std = torch.exp(0.5 * log_var)  
        
        predicted_distribution = D.Normal(mean, std)

        return predicted_distribution, state_emb, action_emb



    def loss(self, batch, epoch):
        
        state, action, effect = batch
        
        predicted_dist, state_emb, action_emb = self.forward(state, action)
        
        nll_loss = -predicted_dist.log_prob(effect).mean()

        nce_loss = self.info_nce_loss(torch.cat([state_emb, action_emb], dim=1))
        nce_loss_state = self.info_nce_loss(state_emb)
        nce_loss_action = self.info_nce_loss(action_emb)

        total_loss = 0.1*nll_loss + 0.1*nce_loss
        return total_loss, nll_loss, nce_loss, predicted_dist.entropy().mean().item(), nce_loss_state, nce_loss_action
 
    def info_nce_loss(self, z, temperature=0.1):
        batch_size = z.shape[0]
        z = F.normalize(z, dim=1)

        sim_matrix = torch.matmul(z, z.T) / temperature
        
        labels = torch.arange(batch_size).to(z.device)

        loss = F.cross_entropy(sim_matrix, labels)

        # Normalize by log(num_negatives)
        loss = loss / math.log(batch_size - 1)

        return loss

    
    def validate_emb(self):
        self.eval()
        tqdm.write(">>> Validating the state")
        with torch.no_grad():
 
            state = torch.tensor([0.07, 0.07, 0.07, 0.0]).unsqueeze(0).to("cuda")
            tqdm.write(f"\tState: {state.cpu().detach().numpy()} -> {self.encode_state(state).detach().cpu().numpy()}")

            state = torch.tensor([0.03, 0.1, 0.02, 0.0]).to("cuda").unsqueeze(0)
            tqdm.write(f"\tState: {state.cpu().detach().numpy()} -> {self.encode_state(state).cpu().detach().numpy()}")

            state = torch.tensor([0.04, 0.05, 0., 1.0]).to("cuda").unsqueeze(0)
            tqdm.write(f"\tState: {state.cpu().detach().numpy()} -> {self.encode_state(state).cpu().detach().numpy()}")

            state = torch.tensor([0.1, 0.1, 0.04, 1.0]).to("cuda").unsqueeze(0)
            tqdm.write(f"\tState: {state.cpu().detach().numpy()} -> {self.encode_state(state).cpu().detach().numpy()}")


    def validate_actions(self):
        self.eval()
        tqdm.write(">>> Validating the actions")
        
        with torch.no_grad():
            actions = torch.randn((2000, 12)).to("cuda")
            actions_dict = set()

            encoded_actions = self.encode_action(actions)

            for i in range(actions.shape[0]):
                actions_dict.add(str((encoded_actions[i].cpu().numpy()>=0.0).astype(int)))

            tqdm.write(f"Min max latent actions: {encoded_actions.min().cpu().numpy()}, {encoded_actions.max().cpu().numpy()}")
            tqdm.write(f"Min max absolute latent actions: {encoded_actions.abs().min().cpu().numpy()}, {encoded_actions.abs().max().cpu().numpy()}")
            tqdm.write(f"\tUnique actions: {actions_dict}")
        



class SIU_Deepsym(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, output_dim, state_latent_dim, action_latent_dim, deterministic=False, activation="ST"):
        super().__init__()
        self.output_dim = output_dim
        self.deterministic=deterministic
        self.state_latent_dim = state_latent_dim
        self.action_latent_dim = action_latent_dim
        self.bottleneck_dim = state_latent_dim + action_latent_dim

        self.state_encoder = nn.Sequential(
            blocks.MLP([state_dim, 128, state_latent_dim], last_layer_norm=True),
            blocks.STLayer()
        )

        self.action_encoder = nn.Sequential(
            blocks.MLP([action_dim, 128, action_latent_dim], last_layer_norm=True),
            blocks.STLayer()
        )

        self.decoder = nn.Sequential(
            blocks.MLP([self.bottleneck_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, output_dim], hiddrop=0.1),
        )
        
    
    def encode_action(self, action):
        return self.action_encoder(action)
    
    def encode_state(self, state):
        return self.state_encoder(state)
    
    def decode(self, state_emb, action_emb):
        effect = self.decoder(torch.cat((state_emb, action_emb), dim=1))

        return effect

    def forward(self, state, action):
        state_emb = self.encode_state(state)  
        
        action_emb = self.encode_action(action)
        
        effect = self.decode(state_emb, action_emb)#state_projected, action_projected)
        
        return effect, state_emb, action_emb



    def loss(self, batch, epoch=None):
        
        state, action, effect = batch
        
        prediction, state_emb, action_emb = self.forward(state, action)
        if epoch is not None:
            print(action_emb.abs().min(dim=0).values, action_emb.abs().max(dim=0).values)
            print(state_emb.abs().min(dim=0).values, state_emb.abs().max(dim=0).values)
        
        return F.mse_loss(prediction, effect)
