#             # 创建一个文件处理器（FileHandler），将日志写入到文件 DCCA.log 中。
#             fh = logging.FileHandler(os.path.join(save_path, f"HiPRS_section_{lambda1}_{lambda2}_{lambda3}.log"), encoding='utf-8')
#             # 设置文件处理器的日志格式，通过之前定义的格式化器（formatter），确保日志按照指定的格式输出到文件中。
#             fh.setFormatter(formatter)
#             # 将之前创建的文件处理器（fh）添加到日志记录器（self.logger）中，使得日志信息可以通过该处理器输出到文件中。
#             self.logger.addHandler(fh)

#             # 通过日志记录器 self.logger 记录模型和优化器的相关信息，并将这些信息输出到指定的日志文件中。
#             # 记录 model1 和 model2 的相关信息
#             self.logger.info("Model 1 (SNP_H1 model):")
#             self.logger.info(self.model_H1)  # 记录第一个子网络的结构和参数

#             self.logger.info("Optimizer for Model:")
#             self.logger.info(self.optimizer)

#             self.logger.info("训练开始......")

#     # fit 方法不会自动被调用，除非显式地调用它。
#     # 调用时，输入的参数分别为train1, train2, val1, val2, test1, test2
#     # checkpoint.model文件用于记录模型结构参数
#     def fit(self, random_state, x1, x2, x3, vx1=None, vx2=None, vx3=None, **kwargs):
#         """

#         x1, x2 are the vectors needs to be make correlated
#         x1、x2是需要关联的向量，对应最初的输入数据（训练集train1, train2，data1[0][0]仅代表图像，不包括标签）
#         x3：样本的AD标签
#         dim=[batch_size, feats]

#         """

#         beta_lassoR_abs = torch.from_numpy(self.beta_lassoR_abs).to(device)

#         checkpoint_H1 = f"checkpoint_H1_random{random_state}_{str(l1)}_{str(l2)}_{str(l3)}.model"

#         # 记录所有epoch的损失，用来画图
#         train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []

#         # 将训练集的输入数据发送到GPU
#         x1 = x1.to(self.device)
#         # x2 = x2.to(self.device)
#         x3 = x3.to(self.device)

#         # 这里指的是训练集的样本量，比如50000
#         data_size = x1.size(0)

#         lassoR_top50snps_indices = [all_snps_id.index(snp) for snp in self.lassoR_top50_snp if snp in all_snps_id]
#         # print("重要SNP的索引为: ", lassoR_top50snps_indices)

#         best_acc = 0
#         best_loss = float('inf')

#         patience = 15
#         no_improve_count = 0
#         best_epoch = 0
#         best_nums = 0

#         # lambda_ = 1e-4
#         # lambda_align = 20
#         # lambda_ld = 1e-4 * 5

#         select_snps = len(self.lassoR_top50_snp) * 2
#         # print(f"lambda_ = {lambda_}，lambda_align = {lambda_align}")
#         # print(f"网络需要选的SNP个数 = {len(self.lassoR_top50_snp)} * 2 = {select_snps}")

#         # ***********开始训练，训练轮次为epoch：0~self.epoch_num-1
#         for epoch in tqdm(range(self.epoch_num), desc="Training Epochs", ncols=100, unit="epoch"):

#             # 记录每轮训练开始的时间
#             epoch_start_time = time.time()

#             # 将模型设置为 训练模式。这行代码通常在训练循环开始时调用。
#             # 这意味着模型中的 BatchNorm 等层会根据训练时的行为进行操作（例如，BatchNorm 会使用当前批次的均值和方差来标准化数据）。
#             self.model_H1.train()

#             # 生成 批量索引（batch_idxs），用于从一个数据集（通常是训练集）中按批次（batch）随机采样数据。
#             # 将训练集索引打乱并分成多个批次，每个批次的大小为 self.batch_size。
#             # drop_last=False：如果最后一个批次的大小不足 batch_size，则保留这个批次。
#             # 整体作用：按bs生成批量索引的列表，比如[[3, 1, 8], [0, 6, 7], [4, 9, 2], [5]]
#             batch_idxs = list(
#                 BatchSampler(RandomSampler(range(data_size)), batch_size=self.batch_size, drop_last=False))

#             batches_train_losses = []
#             batches_train_bce_losses = []
#             batches_train_linput_losses = []
#             batches_train_lalign_losses = []
#             batches_train_lld_losses = []
#             # batches_train_predad_label = []
#             # batches_train_truead_label = []

#             # 对于每一个批次的训练样本
#             for batch_idx in batch_idxs:
#                 # 在 PyTorch 中，每次执行反向传播（backward()）时，计算的梯度会被累加到每个参数的 .grad 属性上，
#                 # 因此在每次进行新的反向传播之前，必须先清空前一次的梯度，否则梯度会不断累加。
#                 self.optimizer.zero_grad()

#                 # 从输入数据 x1 和 x2 中根据 batch_idx 提取出一个批次（batch）的数据
#                 # 此时batch_x1和batch_x2的维度为：800x784(m:每个批次的样本量:bs)(n:特征数:表型数)
#                 batch_x1 = x1[batch_idx, :]
#                 # batch_x2 = x2[batch_idx, :]
#                 batch_x3 = x3[batch_idx, :]

#                 # # Task1: HC (0) vs MCI+AD (1/2)
#                 # target1 = (batch_x3 >= 1).float()
#                 #
#                 # # Task2: MCI (1) vs AD (2)
#                 # target2 = (batch_x3 >= 2).float()
#                 #
#                 # # ---- Task1 权重 ----
#                 # # HC -> task1_weight_HC
#                 # # MCI/AD -> task1_weight_MCI_AD
#                 # weights1 = torch.zeros_like(batch_x3, dtype=torch.float)
#                 # weights1[batch_x3 == 0] = self.task1_weight_HC  # HC
#                 # weights1[batch_x3 == 1] = self.task1_weight_MCI_AD  # MCI
#                 # weights1[batch_x3 == 2] = self.task1_weight_MCI_AD  # AD
#                 #
#                 # # ---- Task2 权重 ----
#                 # # HC/MCI -> task2_weight_HC_MCI
#                 # # AD -> task2_weight_AD
#                 # weights2 = torch.zeros_like(batch_x3, dtype=torch.float)
#                 # weights2[batch_x3 == 1] = self.task2_weight_MCI  # MCI
#                 # weights2[batch_x3 == 2] = self.task2_weight_AD  # AD
#                 # weights2[batch_x3 == 0] = 0.0

#                 # 将该批次的SNP数据输入GenNet，得到H1，shape: [batch_size, 基因个数]
#                 H1_logit = self.model_H1(batch_x1)
#                 w = self.model_H1.w_values
#                 criterion = torch.nn.MSELoss()
#                 L_bce = criterion(H1_logit, batch_x3.double())

#                 # criterion1 = nn.BCEWithLogitsLoss(weight=weights1.view(-1).float())
#                 # loss1 = criterion1(H1_logit.view(-1).float(), target1.view(-1).float())
#                 #
#                 # criterion2 = nn.BCEWithLogitsLoss(weight=weights2.view(-1).float())
#                 # loss2 = criterion2(H1_logit.view(-1).float(), target2.view(-1).float())
#                 #
#                 # L_bce = loss1 + loss2

#                 # 方案4：自适应lassoR权重+度量学习
#                 L_input = self.lambda1 * torch.sum(1 / beta_lassoR_abs * torch.abs(w))

#                 L_align = self.lambda2 * torch.mean((torch.abs(w) - beta_lassoR_abs) ** 2)
#                 #
#                 dist = 1 * torch.norm(torch.abs(w) - beta_lassoR_abs)
#                 # print(f"dist = {dist}")

#                 L_ld = self.lambda3 * torch.sum(torch.sqrt(w[:-1] ** 2 + w[1:] ** 2))

#                 loss = L_bce + L_input + L_align + L_ld

#                 # print(f"第{epoch}轮epoch的batch的总loss为: {loss}，主网络分类loss为: {L_bce}，W的loss为: {L_input}，对齐的loss为: {L_align}，LD的loss为: {L_ld}")

#                 # 将当前批次的loss添加到 train_losses 列表中。
#                 batches_train_losses.append(loss.item())
#                 batches_train_bce_losses.append(L_bce.item())
#                 batches_train_linput_losses.append(L_input.item())
#                 batches_train_lalign_losses.append(L_align.item())
#                 batches_train_lld_losses.append(L_ld.item())

#                 # 反向传播，计算所有模型参数的梯度。
#                 # 首先，数据会通过神经网络进行前向传播（forward），并计算出模型的输出。
#                 # 然后，损失函数会基于模型输出和实际目标计算得到当前的损失值。
#                 # 前两步称为前向传播。
#                 # 接下来为反向传播，1.基于损失函数和网络的计算图，利用链式法则对每个模型参数求导，并存储梯度。
#                 loss.backward()

#                 # self.model_H1.zero_snp_grad()

#                 # 2.使用优化器（如 optimizer.step()）根据这些梯度更新参数。
#                 self.optimizer.step()

#             # 对于每个轮次，求所有批次的loss的平均值
#             epoch_train_loss = np.mean(batches_train_losses)
#             epoch_train_bce_loss = np.mean(batches_train_bce_losses)
#             epoch_train_linput_loss = np.mean(batches_train_linput_losses)
#             epoch_train_lalign_loss = np.mean(batches_train_lalign_losses)
#             epoch_train_lld_loss = np.mean(batches_train_lld_losses)
#             # print(f"epoch_train_loss={epoch_train_loss}，best_loss={best_loss}")

#             # epoch_train_accuracy = (torch.eq(batches_train_predad_label[0],batches_train_truead_label[0]).sum().item() / len(batches_train_truead_label[0])) * 100
#             epoch_train_accuracy = 0

#             # 定义在DCCA.log日志文件中记录每个轮次训练情况的字符串格式
#             # 如：Epoch 1/1 - time: 118.60 - training_loss: -4.8615
#             info_string = "Epoch {:d}/{:d} - time(minute): {:.2f} - training_loss: {:.4f} - L_bce_loss: {:.4f} - L_input_loss: {:.4f} - L_align_loss: {:.4f} - L_ld_loss: {:.4f}"

#             # 记录每轮训练的耗时（分钟）
#             epoch_time = (time.time() - epoch_start_time) / 60

#             # 在DCCA.log日志文件中记录每个轮次训练和验证情况的字符串格式
#             self.logger.info(
#                 info_string.format(epoch + 1, self.epoch_num, epoch_time, epoch_train_loss, epoch_train_bce_loss, epoch_train_linput_loss, epoch_train_lalign_loss, epoch_train_lld_loss))
#             self.logger.info("\n")

#             # 训练终止条件：
#             # 1.w绝对值排序前100名里包括这50个SNP
#             # 2.准确率超过lassoR的准确率(65.23%)
#             with torch.no_grad():
#                 df = pd.DataFrame({
#                     'SNP': all_snps_id,
#                     'Weight': np.abs(self.model_H1.w_values.data.cpu().numpy())
#                 })

#                 top100_snps = df.sort_values(by="Weight", ascending=False).head(select_snps)["SNP"].tolist()

#                 # 转成集合
#                 set_top100_snps = set(top100_snps)
#                 set_lassoR_top50_snps = set(self.lassoR_top50_snp)

#                 num_overlap = len(set_lassoR_top50_snps & set_top100_snps)
#                 # print(
#                 #     f"第{epoch + 1}轮epoch中，Top {select_snps} SNPs 中包含 LassoR 前{len(self.lassoR_top50_snp)} SNPs 的数量: {num_overlap}")

#                 if epoch_train_loss < best_loss:
#                     torch.save(self.model_H1.state_dict(), os.path.join(self.save_path, checkpoint_H1))
#                     best_loss = epoch_train_loss
#                     best_acc = epoch_train_accuracy
#                     no_improve_count = 0
#                     best_epoch = epoch
#                     best_nums = num_overlap
#                 else:
#                     no_improve_count = no_improve_count + 1
#                     if no_improve_count >= patience:
#                         print("在第", epoch + 1, "轮终止训练....")
#                         break

#             train_losses.append(epoch_train_loss)

#         print("本折训练完成。")
#         print("最佳loss为: ", best_loss, "，对应的epoch为: ", best_epoch)

#         self.logger.info("本折训练完成。")

#         return train_losses, train_accuracy


# def set_font_and_alignment(writer, font_name='Times New Roman', font_size=12, alignment_horizontal='left',
#                            alignment_vertical='center'):
#     """
#     设置Excel表格的字体和对齐方式。

#     :param writer: pandas ExcelWriter 对象
#     :param font_name: 字体名称，默认为 'Times New Roman'
#     :param font_size: 字体大小，默认为 12
#     :param alignment_horizontal: 水平对齐方式，默认为 'left'
#     :param alignment_vertical: 垂直对齐方式，默认为 'center'
#     """
#     # 获取写入的工作簿
#     workbook = writer.book

#     # 设置字体和对齐
#     font = Font(name=font_name, size=font_size)
#     alignment = Alignment(horizontal=alignment_horizontal, vertical=alignment_vertical)

#     # 遍历每个工作表并调整字体和对齐
#     for sheet_name in workbook.sheetnames:
#         sheet = workbook[sheet_name]
#         for row in sheet.iter_rows():
#             for cell in row:
#                 cell.font = font
#                 cell.alignment = alignment


# # 把随机五折的每一折的训练、验证、测试集对应原数据集的索引下标保存下来
# # 将 numpy.ndarray 类型直接写入 JSON 文件，但 json.dump() 不支持 ndarray 类型。
# # 需要 先把所有的 np.ndarray 转成 Python 原生的 list。
# def convert_ndarray_to_list(obj):
#     # 如果是 numpy 数组，转换成 list
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     # 如果是 list，对每个元素递归处理
#     elif isinstance(obj, list):
#         return [convert_ndarray_to_list(i) for i in obj]
#     # 如果是字典，每个value递归处理
#     elif isinstance(obj, dict):
#         return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
#     # 如果是其它（int、float、str...），原样返回
#     else:
#         return obj


# # 画出训练和验证的loss和acc图
# def draw_curve(path, train, val, type):
#     epochs = range(1, len(train) + 1)

#     if type == 'loss':
#         plot_curve_label = ['Train_loss', 'Val_loss']
#         plot_y_label = 'Loss'
#         plot_title = 'Training and validation loss'
#     else:
#         plot_curve_label = ['Train_accuracy', 'Val_accuracy']
#         plot_y_label = 'Accuracy (%)'
#         plot_title = 'Training and validation accuracy'

#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, train, label=plot_curve_label[0], color='blue', marker='o')
#     plt.plot(epochs, val, label=plot_curve_label[1], color='orange', marker='o')
#     plt.xlabel('Epoch')
#     plt.ylabel(plot_y_label)
#     plt.title(plot_title)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300)


# if __name__ == '__main__':

#     # 记录开始时间
#     start_time = time.time()

#     ############
#     # Parameters Section
#     # 1.参数部分+数据读取

#     # 将计算设备设置为 GPU，如果有多个 GPU 设备，你可以指定具体的设备编号，例如 'cuda:0'、'cuda:1' 等。
#     # 这里输出的GPU数量为1
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     device = torch.device('cuda')
#     print("Using", torch.cuda.device_count(), "GPUs")
#     gpu_name = torch.cuda.get_device_name(device)
#     print(f"Using GPU: {gpu_name}")

#     # start_idxs = [0, 28580, 53658, 75558, 91054, 106132, 123968, 143839, 157958, 168766, 184055, 204219, 220236, 227399,
#     #               238567, 252229, 265833, 285764, 291095, 309161, 318609, 323034]
#     # end_idxs = [28580, 53658, 75558, 91054, 106132, 123968, 143839, 157958, 168766, 184055, 204219, 220236, 227399,
#     #             238567, 252229, 265833, 285764, 291095, 309161, 318609, 323034, 331631]

#     random_states = [42]
#     start_idxs = [0]
#     end_idxs = [68633]
#     # end_idxs = [104721]
#     threshold = 0.05

#     for random_state in random_states:
#         for j in range(0, len(start_idxs)):
#             start_idx = start_idxs[j]
#             end_idx = end_idxs[j]

#             # 所有用到的数据，比如SNP基因型、先验知识、表型等等，都在gene_snp\snp_feature文件夹下
#             # data_path = os.path.join('D:\PythonCodes\DeepCCA-master\gene_snp', 'snp_feature', 'ADNI_chr19_data')
#             data_path = os.path.join('D:\PythonCodes\DeepCCA-master\gene_snp', 'snp_feature', f'ADNI_brainSNPs_755_{threshold}_data')
#             lassoR_path = os.path.join('D:\PythonCodes\DeepCCA-master\gene_snp', 'LG_GenNet', 'LassoR_results')

#             # 根据给定阈值读取SNP和表型数据
#             # SNP_file_name = os.path.join(data_path, 'ADNI_brainSNPs_755_QC_find_gene_pathway_all_final_' + str(start_idx) + '_' + str(end_idx) + '_final.txt')
#             SNP_file_name = os.path.join(data_path, f'ADNI_brainSNPs_755_QC_{threshold}_train_544_random{random_state}_QC_find_gene_pathway_all_final.txt')

#             SNP, truead, SNP_to_topology, IID = three_load_SNP(SNP_file_name)

#             # ols_weights = pd.read_excel(os.path.join(data_path, 'chr19_QC_snps_base_info_final.xlsx')).iloc[start_idx:end_idx].reset_index(drop=True)
#             ols_weights = pd.read_excel(os.path.join(data_path, f'ADNI_brainSNPs_755_QC_{threshold}_base_info_final_ld.xlsx')).iloc[start_idx:end_idx].reset_index(drop=True)

#             if start_idx == 0 and end_idx == 80454:
#                 topology = pd.read_csv('ADNI_680_p0.01_train_544_random42_QC_find_gene_pathway_all_final_topology.csv')
#             else:
#                 topology = False

#             # 模型最佳参数保存目录为gene_snp\model\GenNet_best_model
#             # model_path = os.path.join(os.getcwd(), 'GenNet_skip_all_dfs_best_model', 'no_pcc_threshold', 'all')
#             model_path = rf'D:\PythonCodes\DeepCCA-master\gene_snp\LG_GenNet\three_classes_pcc\random{random_state}_HiPRS_allchr_l123_results_{threshold}\prs_best_model'
#             # 如果目录不存在，则递归创建
#             os.makedirs(model_path, exist_ok=True)
#             # 有关模型的训练和验证曲线图、以及SNP和基因对应的权重，保存到该目录
#             # result_path = os.path.join(os.getcwd(), 'GenNet_skip_all_dfs_train_result', 'no_pcc_threshold', 'all')
#             result_path = rf'D:\PythonCodes\DeepCCA-master\gene_snp\LG_GenNet\three_classes_pcc\random{random_state}_HiPRS_allchr_l123_results_{threshold}\prs_train_result'
#             # 如果目录不存在，则递归创建
#             os.makedirs(result_path, exist_ok=True)

#             # 保存所有SNP对应的染色体编号
#             snp_chr = {}
#             # ADNI_chr19_755_QC_3.bim
#             # ADNI_brainSNPs_755_QC_3.bim
#             with open(os.path.join(data_path, 'ADNI_brainSNPs_755_QC_3.bim')) as f:
#                 for line in f:
#                     chr, snp = line.strip().split()[0], line.strip().split()[1]
#                     snp_chr[snp] = chr

#             # 只对SNP标准化
#             train_snp = SNP.to_numpy()
#             train_snp = (train_snp - np.mean(train_snp, axis=0)) / np.std(train_snp, axis=0, ddof=1)
#             train_snp = torch.from_numpy(train_snp).double()
#             train_truead = torch.tensor(truead.values)
#             print("训练集样本数为: ", len(train_snp))

#             print("样本的AD状态分布为: ")
#             counts = torch.bincount(train_truead.view(-1).long())
#             for cls, count in enumerate(counts):
#                 print(f"Class {cls}: {count} samples")
#             num_classes = 1

#             # AD类权重 和 HC类权重
#             hc = counts[0].float()
#             mci = counts[1].float()
#             ad = counts[2].float()
#             total = counts.sum().float()

#             # Task1：区分 HC vs (MCI + AD)
#             task1_weight_HC = total / (2 * hc)
#             task1_weight_MCI_AD = total / (2 * (mci + ad))

#             # Task2：区分 MCI vs AD
#             task2_weight_MCI = total / (2 * mci)
#             task2_weight_AD = total / (2 * ad)
#             print(f"任务 1 为区分 HC vs (MCI + AD)，其中 HC 的权重为{task1_weight_HC}，(MCI + AD) 的权重为{task1_weight_MCI_AD}")
#             print(f"任务 2 为区分 MCI vs AD，其中 MCI 的权重为{task2_weight_MCI}，AD 的权重为{task2_weight_AD}")

#             SNP_to_topology = SNP_to_topology.iloc[:-1, :]

#             SNP_nums = SNP.shape[1]
#             learning_rate = 1e-3
#             epoch_num = 10000
#             batch_size = SNP.shape[0]
#             reg_par = 1e-5

#             # 选出lassoR的权重绝对值降序前50名SNP
#             topsnps = 200
#             lassoR_df = pd.read_excel(os.path.join(lassoR_path, f'ADNI_brainSNPs_755_QC_{threshold}_train_544_random{random_state}_QC_lassoR_9_1e-8.xlsx'))
#             top50_snps = lassoR_df.sort_values(by="regression_weight", ascending=False).head(topsnps)["snp"].tolist()
#             all_snps_id = SNP.columns.tolist()
#             lassoR_acc = 0.6523 * 100
#             beta_lassoR_abs = lassoR_df['regression_weight'].values * 1e5
#             n_select = SNP_nums

#             # 计算当前批次的起始和结束索引
#             start_idx = 0 * n_select
#             end_idx = min((0 + 1) * n_select, SNP_nums)
#             train_snp_select = train_snp[:, start_idx:end_idx]

#             if topology is False:
#                 train_snp_select_topology = Create_topology(SNP_to_topology, start_idx, end_idx, None, snp_chr)
#             else:
#                 train_snp_select_topology = topology

#             # train_snp_select_topology = Create_topology(SNP_to_topology, start_idx, end_idx, None, snp_chr)

#             snp_select_nums = int(train_snp_select_topology['layer0_node'].max() + 1)
#             Gene_select_nums = int(train_snp_select_topology['layer1_node'].max() + 1)
#             Pathway_select_nums = int(train_snp_select_topology['layer2_node'].max() + 1)
#             Features_with_ADpred_nums = Pathway_select_nums + Gene_select_nums + snp_select_nums
#             print("主网络用于预测AD状态的特征数(所有通路 + 所有基因)为: ", Features_with_ADpred_nums, "共包括", Gene_select_nums, "个基因，",
#                   Pathway_select_nums, "个通路")

#             # lambda1s = [1e-5, 2e-5, 5e-5, 1e-4]  # 自适应 L1
#             # lambda2s = [5, 10, 20, 50]  # lassoR 对齐
#             # lambda3s = [1e-5, 2e-5, 5e-5, 1e-4]  # LD 平滑
#             # lambda1s = [5e-5, 1e-4]  # 自适应 L1
#             # lambda2s = [10, 20, 50]  # lassoR 对齐
#             # lambda3s = [2e-5, 5e-5, 1e-4]  # LD 平滑
#             lambda1s = [5e-5]  # 自适应 L1
#             lambda2s = [5, 10, 20, 50]  # lassoR 对齐
#             lambda3s = [1e-5, 2e-5, 5e-5, 1e-4]  # LD 平滑

#             for l1, l2, l3 in product(lambda1s, lambda2s, lambda3s):

#                 print("网络的各项基础参数为: ")
#                 print(
#                     f"epoch: {epoch_num}, bs: {batch_size}, lr: {learning_rate}, lambda1: {l1}, lambda2: {l2}, lambda3: {l3}")

#                 # 5.构建网络模型，初始化参数

#                 model_SNP = GenNet(input_size=SNP_nums, hidden_size=Features_with_ADpred_nums, output_size=num_classes,
#                                    topology=train_snp_select_topology, device=device)

#                 solver_phe = Solver_PheWAS(model_SNP, epoch_num, batch_size, learning_rate, reg_par, l1, l2, l3,
#                                            task1_weight_HC, task1_weight_MCI_AD, task2_weight_MCI, task2_weight_AD,
#                                            top50_snps,
#                                            all_snps_id, lassoR_acc, beta_lassoR_abs, model_path, device=device)

#                 # 6.开始训练
#                 st1 = time.time()
#                 phe_train_losses, phe_train_acc = solver_phe.fit(random_state, train_snp_select, False, train_truead)
#                 en1 = time.time()
#                 print(f"训练时间为: {(en1 - st1) / 60} 分钟")

#                 model_SNP.load_state_dict(
#                     torch.load(os.path.join(model_path,
#                                             f'checkpoint_H1_random{random_state}_{str(l1)}_{str(l2)}_{str(l3)}.model')),
#                     strict=False)
#                 snp_dfs_weights = model_SNP.w_values.data.cpu().numpy()
#                 train_snp_select = train_snp_select.to(device)

#                 # 构建三个层级的weights
#                 coordinates_snp = ols_weights.copy()
#                 coordinates_snp.iloc[:, 0] = lassoR_df.iloc[:, 0]
#                 coordinates_snp.columns.values[0] = "raw_index"
#                 pcc_position = coordinates_snp.columns.get_loc('pcc_abs')
#                 coordinates_snp.insert(pcc_position + 1, 'dfs_weight', np.abs(snp_dfs_weights))
#                 coordinates_snp.insert(pcc_position + 2, 'regression_weight', beta_lassoR_abs)

#                 # 将 DataFrame 写入 GenNet_train_result/指定阈值/importance_snp_gene.xlsx
#                 output_excel = os.path.join(result_path,
#                                             f'w_importance_random{random_state}_{str(l1)}_{str(l2)}_{str(l3)}.xlsx')
#                 if not os.path.exists(output_excel):
#                     with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
#                         coordinates_snp.to_excel(writer, sheet_name='snp', index=False)
#                         # 在写入数据后调用 set_font_and_alignment 函数
#                         set_font_and_alignment(writer, font_name='Times New Roman', font_size=12,
#                                                alignment_horizontal='left', alignment_vertical='center')
#                 print(f"数据已成功写入 {output_excel}")

#     end_time = time.time()
#     # 计算执行时间
#     elapsed_time = (end_time - start_time) / 60
#     print(f"执行时间为: {elapsed_time} 分钟")
