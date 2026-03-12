import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
import json

# 打印调试信息
print("Starting Hyper-ZPD-Net...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 数据集路径
DATASET_PATH = "dataset/Assistment"

class EducationalDataset(Dataset):
    def __init__(self, data_dir, max_seq_length=100):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.student_data = self.load_data()
        self.skill_to_id, self.id_to_skill = self.build_skill_vocab()
        self.num_skills = len(self.skill_to_id)
    
    def load_data(self):
        student_data = {}
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv') and 'student_log' in f]
        
        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                student_id = row['ITEST_id']
                skill = row['skill']
                correct = int(row['correct'])
                start_time = row['startTime']
                time_taken = row['timeTaken']
                
                if student_id not in student_data:
                    student_data[student_id] = []
                
                student_data[student_id].append({
                    'skill': skill,
                    'correct': correct,
                    'start_time': start_time,
                    'time_taken': time_taken
                })
        
        # 按时间排序
        for student_id in student_data:
            student_data[student_id].sort(key=lambda x: x['start_time'])
        
        return student_data
    
    def build_skill_vocab(self):
        skills = set()
        for student_id in self.student_data:
            for item in self.student_data[student_id]:
                skills.add(item['skill'])
        
        skill_to_id = {skill: i for i, skill in enumerate(skills)}
        id_to_skill = {i: skill for skill, i in skill_to_id.items()}
        return skill_to_id, id_to_skill
    
    def __len__(self):
        return len(self.student_data)
    
    def __getitem__(self, idx):
        student_id = list(self.student_data.keys())[idx]
        student_seq = self.student_data[student_id]
        
        # 截取最近的 max_seq_length 个交互
        student_seq = student_seq[-self.max_seq_length:]
        
        # 填充到 max_seq_length
        seq_length = len(student_seq)
        padding_length = self.max_seq_length - seq_length
        
        skills = []
        corrects = []
        time_taken = []
        time_diffs = []
        
        # 计算时间差
        prev_time = None
        for item in student_seq:
            skills.append(self.skill_to_id[item['skill']])
            corrects.append(item['correct'])
            time_taken.append(item['time_taken'])
            
            if prev_time is not None:
                time_diff = item['start_time'] - prev_time
                time_diffs.append(time_diff)
            else:
                time_diffs.append(0)
            prev_time = item['start_time']
        
        # 填充
        if padding_length > 0:
            skills = [0] * padding_length + skills
            corrects = [0] * padding_length + corrects
            time_taken = [0] * padding_length + time_taken
            time_diffs = [0] * padding_length + time_diffs
        
        return {
            'student_id': student_id,
            'skills': torch.tensor(skills, dtype=torch.long),
            'corrects': torch.tensor(corrects, dtype=torch.float),
            'time_taken': torch.tensor(time_taken, dtype=torch.float),
            'time_diffs': torch.tensor(time_diffs, dtype=torch.float),
            'seq_length': seq_length
        }

class TemporalDecayHHC(nn.Module):
    def __init__(self, num_skills, hidden_dim, num_layers=3, tau=3.0):
        super(TemporalDecayHHC, self).__init__()
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.tau = tau
        
        # 技能嵌入
        self.skill_embedding = nn.Embedding(num_skills, hidden_dim)
        
        # 时间衰减参数
        self.time_decay = nn.Parameter(torch.tensor([tau], dtype=torch.float))
        
        # 超图卷积层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def temporal_decay_factor(self, time_diffs):
        # 计算时间衰减因子
        return torch.exp(-time_diffs / self.time_decay)
    
    def forward(self, skills, time_diffs):
        # 获取技能嵌入
        skill_emb = self.skill_embedding(skills)
        
        # 计算时间衰减因子
        decay_factors = self.temporal_decay_factor(time_diffs).unsqueeze(-1)
        
        # 应用时间衰减
        decay_emb = skill_emb * decay_factors
        
        # 超图卷积
        for conv in self.conv_layers:
            decay_emb = self.relu(conv(decay_emb))
        
        return decay_emb

class CognitiveStateCGU(nn.Module):
    def __init__(self, hidden_dim, alpha=0.2, beta=0.8):
        super(CognitiveStateCGU, self).__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        
        # 门控参数
        self.gate_weight = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # 更新参数
        self.update_weight = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def cognitive_distance(self, s_t, c_t):
        # 计算认知距离
        norm_s = torch.norm(s_t, dim=-1, keepdim=True)
        norm_c = torch.norm(c_t, dim=-1, keepdim=True)
        dot_product = torch.sum(s_t * c_t, dim=-1, keepdim=True)
        distance = dot_product / (norm_s * norm_c + 1e-8)
        return distance
    
    def forward(self, s_t, c_t):
        # 计算认知距离
        d_cognitive = self.cognitive_distance(s_t, c_t)
        
        # 计算门控值
        gate_input = torch.cat([s_t, c_t, d_cognitive], dim=-1)
        g_gate = self.sigmoid(self.gate_weight(gate_input) + self.gate_bias)
        
        # ZPD 门控
        zpd_mask = (d_cognitive >= self.alpha) & (d_cognitive <= self.beta)
        g_zpd = g_gate * zpd_mask.float()
        
        # 计算更新
        update_input = torch.cat([s_t, c_t], dim=-1)
        update = self.tanh(self.update_weight(update_input) + self.update_bias)
        
        # 更新认知状态
        s_t_plus_1 = (1 - g_zpd) * s_t + g_zpd * update
        
        return s_t_plus_1, d_cognitive

class CurriculumGradientARL(nn.Module):
    def __init__(self, hidden_dim, num_skills):
        super(CurriculumGradientARL, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_skills = num_skills
        
        # 策略网络
        self.policy_network = nn.Linear(hidden_dim, num_skills)
        
        # 对手网络
        self.adversary_network = nn.Linear(hidden_dim, num_skills)
        
        # 价值网络
        self.value_network = nn.Linear(hidden_dim, 1)
    
    def forward(self, s_t):
        # 策略网络输出
        policy_logits = self.policy_network(s_t)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # 对手网络输出
        adversary_logits = self.adversary_network(s_t)
        adversary_probs = F.softmax(adversary_logits, dim=-1)
        
        # 价值网络输出
        value = self.value_network(s_t)
        
        return policy_probs, adversary_probs, value
    
    def select_action(self, s_t, epsilon=0.1):
        # epsilon-greedy 策略
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_skills)
        else:
            with torch.no_grad():
                policy_probs, _, _ = self.forward(s_t)
                return torch.argmax(policy_probs, dim=-1).item()

class HyperZPDNet(nn.Module):
    def __init__(self, num_skills, hidden_dim=256, num_layers=3, tau=3.0):
        super(HyperZPDNet, self).__init__()
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        
        # T-HHC 模块
        self.t_hhc = TemporalDecayHHC(num_skills, hidden_dim, num_layers, tau)
        
        # CS-CGU 模块
        self.cs_cgu = CognitiveStateCGU(hidden_dim)
        
        # CG-ARL 模块
        self.cg_arl = CurriculumGradientARL(hidden_dim, num_skills)
        
        # 预测网络
        self.predictor = nn.Linear(hidden_dim, 1)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, skills, time_diffs, corrects):
        batch_size, seq_length = skills.shape
        
        # 初始化认知状态
        s_t = torch.zeros(batch_size, self.hidden_dim).to(skills.device)
        
        # 存储预测结果和认知距离
        predictions = []
        cognitive_distances = []
        
        for t in range(seq_length):
            # 提取当前时间步的输入
            skill_t = skills[:, t]
            time_diff_t = time_diffs[:, t]
            correct_t = corrects[:, t]
            
            # T-HHC 模块
            c_t = self.t_hhc(skill_t, time_diff_t)
            
            # CS-CGU 模块
            s_t, d_cognitive = self.cs_cgu(s_t, c_t)
            
            # 预测
            pred = self.sigmoid(self.predictor(s_t))
            
            # 存储结果
            predictions.append(pred)
            cognitive_distances.append(d_cognitive)
        
        # 转换为张量
        predictions = torch.cat(predictions, dim=1)
        cognitive_distances = torch.cat(cognitive_distances, dim=1)
        
        return predictions, cognitive_distances, s_t
    
    def recommend(self, s_t, epsilon=0.1):
        # 推荐下一个技能
        return self.cg_arl.select_action(s_t, epsilon)

def train(model, train_loader, val_loader, epochs=100, learning_rate=0.0005, weight_decay=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        for batch in train_loader:
            skills = batch['skills'].to(device)
            corrects = batch['corrects'].to(device)
            time_taken = batch['time_taken'].to(device)
            time_diffs = batch['time_diffs'].to(device)
            seq_lengths = batch['seq_length']
            
            optimizer.zero_grad()
            predictions, _, _ = model(skills, time_diffs, corrects)
            
            # 计算损失
            loss = 0.0
            for i, seq_len in enumerate(seq_lengths):
                loss += criterion(predictions[i, -seq_len:], corrects[i, -seq_len:])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 收集预测和标签
            for i, seq_len in enumerate(seq_lengths):
                train_predictions.extend(predictions[i, -seq_len:].detach().cpu().numpy())
                train_labels.extend(corrects[i, -seq_len:].detach().cpu().numpy())
        
        # 计算训练指标
        train_auc = roc_auc_score(train_labels, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(train_labels, train_predictions))
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                skills = batch['skills'].to(device)
                corrects = batch['corrects'].to(device)
                time_taken = batch['time_taken'].to(device)
                time_diffs = batch['time_diffs'].to(device)
                seq_lengths = batch['seq_length']
                
                predictions, _, _ = model(skills, time_diffs, corrects)
                
                # 计算损失
                for i, seq_len in enumerate(seq_lengths):
                    val_loss += criterion(predictions[i, -seq_len:], corrects[i, -seq_len:])
                
                # 收集预测和标签
                for i, seq_len in enumerate(seq_lengths):
                    val_predictions.extend(predictions[i, -seq_len:].detach().cpu().numpy())
                    val_labels.extend(corrects[i, -seq_len:].detach().cpu().numpy())
        
        # 计算验证指标
        val_auc = roc_auc_score(val_labels, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_predictions))
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_hyper_zpd_net.pth')
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train AUC: {train_auc:.4f}, Train RMSE: {train_rmse:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val AUC: {val_auc:.4f}, Val RMSE: {val_rmse:.4f}')
        print('-' * 50)
    
    return best_auc

def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    test_predictions = []
    test_labels = []
    cognitive_distances = []
    
    with torch.no_grad():
        for batch in test_loader:
            skills = batch['skills'].to(device)
            corrects = batch['corrects'].to(device)
            time_taken = batch['time_taken'].to(device)
            time_diffs = batch['time_diffs'].to(device)
            seq_lengths = batch['seq_length']
            
            predictions, distances, _ = model(skills, time_diffs, corrects)
            
            # 收集预测和标签
            for i, seq_len in enumerate(seq_lengths):
                test_predictions.extend(predictions[i, -seq_len:].detach().cpu().numpy())
                test_labels.extend(corrects[i, -seq_len:].detach().cpu().numpy())
                cognitive_distances.extend(distances[i, -seq_len:].detach().cpu().numpy())
    
    # 计算指标
    auc = roc_auc_score(test_labels, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
    
    # 计算 ZPD 合规率
    zpd_compliance = np.mean([1 if 0.2 <= d <= 0.8 else 0 for d in cognitive_distances])
    
    return auc, rmse, zpd_compliance

def main():
    # 加载数据集
    dataset = EducationalDataset(DATASET_PATH)
    
    # 分割数据集
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 初始化模型
    model = HyperZPDNet(dataset.num_skills)
    
    # 训练模型
    print("Training Hyper-ZPD-Net...")
    best_auc = train(model, train_loader, val_loader)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_hyper_zpd_net.pth'))
    
    # 评估模型
    print("Evaluating Hyper-ZPD-Net...")
    auc, rmse, zpd_compliance = evaluate(model, test_loader)
    
    # 输出结果到文件
    with open('model_results.txt', 'w') as f:
        f.write(f"Model: Hyper-ZPD-Net\n")
        f.write(f"Best Validation AUC: {best_auc:.4f}\n")
        f.write(f"Test AUC: {auc:.4f}\n")
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"ZPD-Compliance Rate: {zpd_compliance:.4f}\n")
        f.write(f"Number of skills: {dataset.num_skills}\n")
        f.write(f"Dataset size: {len(dataset)}\n")
        f.write(f"Training size: {train_size}\n")
        f.write(f"Validation size: {val_size}\n")
        f.write(f"Test size: {test_size}\n")
    
    print("Results saved to model_results.txt")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"ZPD-Compliance Rate: {zpd_compliance:.4f}")

if __name__ == "__main__":
    main()
