# How to understand transferability

## A Unified Definition for Transferability
Here, we give several definitions related to transferability, and the summary of these notations and their descriptions used in this survey can be found in Table \ref{tab:notations}.
Given a dataset $\mathcal{D}_S=\{x_1, x_2, \ldots, x_n\}$ sampled i.i.d. from a source distribution $D_S$, a learning {task} is to learn the labeling function $f$ to map the data on the 
input space $\mathcal{X}$ to the label space $\mathcal{Y}$. 
The goal is to find a hypothesis function from the set of all possible functions that can best estimate the function $f$. 
% The goal is to find a single hypothesis function 
% $h: \mathcal{X} \to \mathcal{Y}$ from the hypothesis space $\mathcal{H}$ of all possible functions that can best estimate the function $f$. 
% The learning objective for the source task is to minimize the generalization error $\mathbf{E}_{\mathbf{x}\sim D_S}[\mathcal{L}(h(\mathbf{x}),f(\mathbf{x}))]$.
In practical supervised learning, the labeling function $f$ is often implemented through a combination of a feature extractor $g(x):\mathcal{X}\rightarrow\mathcal{Z}$ and a task head $h(g(x)):\mathcal{Z}\rightarrow\mathcal{Y}$. 



% Please add the following required packages to your document preamble:
% \usepackage{booktabs}
\begin{table}[]
\begin{tabular}{p{2cm} p{4cm} | p{2cm} p{4cm}} 
% \begin{tabular}{@{}ll|ll@{}}
\toprule
\textbf{Symbol} & \textbf{Description}                  & \textbf{Symbol}       & \textbf{Description}                                    \\ \midrule
$f$             & Decision function                     & $H(A)$                & Entropy of event A                                      \\
$h$             & Hypothesis function                   & $\star$               & Placeholder for S and T                                 \\
$\theta$        & Model parameters                      & $\lozenge$            & Placeholder for source and target corresponding $\star$ \\
$\mathcal{H}$   & Hypothesis space                      & $D_\star$             & Distribution of $\lozenge$ data                         \\
$k$             & Kernel function                       & $\mathcal{D}_\star$   & The $\lozenge$ dataset                                  \\
$\mathbb{H}$    & Hilbert Space                         & $\mathcal{T}_\star$   & The $\lozenge$ task                                     \\
$d$             & A certain distance                    & $\mathcal{X}_\star$   & The $\lozenge$ instance space                           \\
$\mathcal{L}$   & Loss function                         & $\mathcal{Y}_\star$   & The $\lozenge$ label space                              \\
$\phi$          & Feature extractor                     & $N_\star$             & Number of $\lozenge$ data                               \\
$p$             & Probability density function          & $X_\star$             & The $\lozenge$ instance set                             \\
$\mathbb{E}[X]$ & Expectation of random variable \(X\)  & $Y_\star$             & Label set corresponding $X_\star$                       \\
$\text{Var}(X)$ & Variance of \(X\)                     & $x_\star$             & an instance from $\lozenge$ dataset                     \\
$\pi(x,y)$      & Joint distribution of \(x\) and \(y\) & $y_\star$             & Label corresponding $x_\star$                           \\
$P(A)$          & Probability of event \(A\)            & $Trf(S\rightarrow T)$ & Transferability from S to T                             \\ \bottomrule
\end{tabular}
\caption{A summary of notations used in this survey.}
\label{tab:notations}
\end{table}


Transferring important source knowledge to target tasks is a common topic in various learning paradigms, while different learning problems transfer different types of knowledge. 
In a classical supervised learning setting, a model pre-trained on a source dataset can be fine-tuned on a target task, thus reusing the entire model architecture and its learned parameters.
In other cases, only task knowledge is carried over, for example, retaining a pre-trained feature extractor and learning a new task head, or conversely, retaining the task head while adapting it to a new feature representation; or reusing the labeled source data as important prior knowledge to assist the training for the target task.
Some other lately popular learning paradigms reveal novel knowledge types when examined through the lens of “what knowledge can be transferred.” 
For instance, in prompt learning, 
instead of updating the model weights, the pre-trained model is kept frozen while being instructed towards target tasks via learning the lightweight embeddings, named prompt.
Similarly, in reinforcement learning, the previously learned policy or transition pattern can be reused from one environment to another, thus transferring the underlying behavioral knowledge across tasks. 
% architectural knowledge, task knowledge, data knowledge, instruction knowledge, behavioral knowledge

Transfer learning occurs whenever knowledge acquired in one task can be reused to assist the learning of a target task.
The heterogeneity of transferable knowledge across multiple learning paradigms necessitates a unified framework. 
In this survey, we refer to these varied transferable elements—such as model parameters, model architectures, task-specific modules, data distributions, or learned strategies—as different {\bf knowledge modality}.
Furthermore, we define {\bf transferability} as a quantitative measure of how effectively the knowledge in a given source task can be reused in a new task. 
We consider transferability at two levels of granularity: {\bf task-level} transferability, which evaluates the impact of the transferred knowledge on the overall target task performance, whereas {\bf instance-level} transferability assesses its influence on individual data instances. 
Through this unified perspective of transferable knowledge type and measurement granularity, this survey offers a novel understanding of how transferability fits in diverse learning paradigms. %allowing people to systematically unify a broad spectrum of learning problems within a coherent analytical framework.


% A supervised learning task consists of learning a model $f$ that maximizes $p(y|x)$ over dataset $D=\{X_n,Y_n\}$. 
% In prompt learning, a prompt $p$ is learned with fixed model $f$ to maximize $p(y|p, x)$ over dataset $D=\{P_n,X_n,Y_n\}$. In reinforcement learning, a policy $\pi$ in a given Markov decision process $(S,A,r(s,a),P_{sa},\pi)$ is learned to maximize the expected cumulative rewards from interaction data $D=\{S_n,A_n,S'_n,r_n\}$.  
% In all these learning problems, transfer learning can be achieved whenever one or more components in a source task, carrying the source knowledge, are used to optimize the learning goal of another target task. Here we refer to these transferable components as different ``{\bf knowledge modalities}", which could include the dataset, model weights, features, prompts, adversarial examples, etc. {\bf Transferability} is a quantitative measure that evaluates how effective the knowledge transfer is, in relation to the entire target task, or to a single data instance. We refer to them as {\bf task-level} and {\bf instance-level}  transferability, respectively. 
% Other granularities of transferability estimation may also be relevant in specific knowledge transfer scenarios. 

% \begin{figure}[htbp]
%     \centering
%     \includegraphics[width=0.8\linewidth]{pictures/RL.jpg}
%     \caption{Reinforcement Learning (RL) is a learning paradigm where an agent interacts with an environment to learn optimal behavior through trial and error. As shown in the diagram, at each time step $ t $, the agent observes a state $s_t$, takes an action $a_t$, and receives a reward $r_{t+1}$ and a new state $s_{t+1}$ from the environment. The agent aims to learn a policy that maximizes the expected cumulative reward over time.
% }
%     \label{fig:enter-label}
% \end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\linewidth]{pictures/prompt-learn-correction.png}
    \caption{(a) Both model backbone and task head are tunable. (b) (c) The backbone of the model remains frozen, only task head is tunable. (d) An overview of prompt-based transfer learning. The backbone of the model remains frozen, while the task-specific head and prompt modules are tunable.
}
    \label{fig:enter-label}
\end{figure}
% \begin{figure}[htbp]
%     \centering
%     \includegraphics[width=0.8\linewidth]{pictures/SL.jpg}
%     \caption{Supervised learning framework. The model is trained using labeled data to learn the mapping from input features to output labels, enabling it to predict labels for new data.
% }
%     \label{fig:enter-label}
% \end{figure}

% In the following text, we will uniformly use the subscript $s$ to denote data or models from the source domain and the subscript $t$ to denote data or models from the target domain.




% \subsection*{Notation List}

% \begin{table}[t]
% \begin{tabular}{p{2cm} p{4cm} | p{2cm} p{4cm}} 
%     % \toprule
%     % \textbf{Symbol} & \textbf{Description} \\ 
%     % \midrule
%     % \endfirsthead
    
%     \toprule
%     \textbf{Symbol} & \textbf{Description} & \textbf{Symbol} & \textbf{Description} \\ 
%     \midrule
%     % \endhead
 
%     $f$ & Decision function \\ 
%     $h$ & Hypothesis function \\ 
%     $\theta$ & Model parameters \\ 
%     $\mathcal{H}$ & Hypothesis space \\    
%     $k$ & Kernel function \\
%     $\mathbb{H}$ & Hilbert Space \\
%     $d$ & A certain distance \\ 
%     $\mathcal{L}$ & Loss function \\ 
%     $\phi$ & Feature extractor \\ 
%     $p$ & Probability density function \\ 
%     $\mathbb{E}[X]$ & Expectation of random variable \(X\) \\ 
%     $\text{Var}(X)$ & Variance of \(X\) \\ 
%     $\pi(x,y)$ & Joint distribution of \(x\) and \(y\) \\ 
%     $P(A)$ & Probability of event \(A\) \\ 
%     $H(A)$ & Entropy of event A  \\ 
%     $\star$ & Placeholder for S and T \\
%     $\lozenge$ & Placeholder for source and target corresponding $\star$ \\
%     $D_\star$ & Distribution of $\lozenge$ data \\ 
%     $\mathcal{D}_\star$ & The $\lozenge$ dataset \\ 
%     $\mathcal{T}_\star$ & The $\lozenge$ task \\ 
%     $\mathcal{X}_\star$ & The $\lozenge$ instance space \\ 
%     $\mathcal{Y}_\star$ & The $\lozenge$ label space \\ 
%     $N_\star$ & Number of $\lozenge$ data\\  
%     $X_\star$ & The $\lozenge$ instance set\\
%     $Y_\star$ & Label set corresponding $X_\star$\\
%     $x_\star$ & an instance from $\lozenge$ dataset\\
%     $y_\star$ & Label corresponding $x_\star$\\
%     $Trf(S\rightarrow T)$ & Transferability from S to T  \\ 
%     \bottomrule
% \end{tabular}
% \end{table}





\begin{definition}[Transferability]
   The {\em transferability} between the source learning task $\mathcal{T}_s$ and target learning task $\mathcal{T}_t$ is defined as the effectiveness of transfering the source knowledge ${K}_s$ to the target, denoted as $Trf( {K}_s, {K}_t, E)$, where $K_s$ is the source knowledge modality that is being transferred, $K_t$ is the target knowledge modality used in computing transferability and $E\subseteq  D_t$ is the evaluation target.  
\end{definition}

%{\em Remarks}
%\begin{itemize} 
% Here we use “task” in the same abstract sense as a “learning problem”, which could encompass the definition of a  distribution (e.g. for supervised learning, a joint distribution between input and labels), a dataset or a model trained on a specific task. 
%\item $E$ is 
%\end{itemize}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{pictures/framework-0416.png}
    \caption{Transferability estimation between source and target model modality, which transferability is computed over different granuality (e.g. tasks or instances).}
    \label{fig:enter-label}
\end{figure}


% ### Knowledge Modalities}

% Before discussing taxonomy of transferability, we will first classify transfer learning paradigms by the knowledge modality being transferred.

% {\bf Model transfer. }This modality referrs to certain model components, such as weights, features, attention maps etc, is being transferred from a source task to a target task. The transferred weights can be copied directly to the target model, as in zero-shot learning, or adapted via fine tuning, model distillation or other adaption techniques.   A model may consists of multiple knowledge modalities \cite{wulfmeier2023foundations}, such as in reinforcement learning, the MDP contains the reward model, dynamics model and policy model,  and any subset of modalities can be transfered to enhance the learning on the target task. 

% {\bf Dataset transfer. }The dataset  modality is involved when we utilize the data from another task (sometimes also referred as domain)   for training the target model such as in domain adapation. The absence of source model implies that the transfer learning algorithm is tasked with extracting the source knowledge that is useful for the target task.   

% {\bf Prompt transfer and Adversarial example transfer}:   In prompt learning and adversarial attack models, the prompts and the adversarial examples are also learning component with respect to a task. Therefore, we can consider them to be special cases in model transfer, even though funtionally they are different from a model.
% …. 

 
  % (Note that we do not discuss the specific methodology for transferability estimation here. They will be introduced in detail  in the next chapter. )

## A Taxonomy of Transferability
Using the unified transferability definition, we can group transferability problems based on the knowledge modalities they are measuring, the granuality of evaluation, and the stage where transferability is computed. In this section, we will discuss each of these categorizations, focusing on how they are related to different transferability estimation mechanisms and applications. 
### Measuring Transferability for Different Knowledge Modalities

% 表格里删掉prompt
% \begin{table}[t]
% \sf
% \begin{tabular}{l p{3.6cm} p{3.6cm} p{3.6cm}}
% \toprule
%   & Model transferability	&Dataset transferability	& Prompt transferability\\
%   \midrule\\
% Task-level &	 $Trf( ( f_s [,\mathcal{D}_s]), \mathcal{D}_t, \mathcal{D}_t)$ \quad  Application: pretrained model selection	 & $Trf(\mathcal{D}_s, \mathcal{D}_t,\mathcal{D}_t)$ \quad  Application: task embedding, dataset similarity, domain selection	& $Trf((\delta_s [, \mathcal{D}_s]),(f_t,\mathcal{D}_t), \mathcal{D}_t)$\quad  Application: Adversarial example transferability analysis \cite{waseda2023closer}, prompt tuning \cite{su2021transferability}   \\
% Instance-level	& $Trf((f_s [,\mathcal{D}_s]),  \mathcal{D}_t , (x_t,y_t))$ \quad 
% Application: Instance-adaptive transfer learning &	$Trf(\mathcal{D}_s,[\mathcal{D}_t], (x_t,y_t)) $ \quad 
% Application: OOD detection, active sample selection in domain adaptation	 & \\
% \bottomrule
% \end{tabular}
% \caption{A taxonomy of transferability based on the knowledge modality being transferred (column) and granuality of transferability evaluation (rows)}
% \end{table}

% Please add the following required packages to your document preamble:
% \usepackage{booktabs}
\begin{table}[]
\begin{tabular}{l p{5.5cm} p{5.5cm}}
\toprule
               & Model transferability                                                                                                          & Dataset transferability                                                                                                                        \\ \midrule 

Task-level     & $Trf( ( f_s [,\mathcal{D}_s]), \mathcal{D}_t, \mathcal{D}_t)$ \quad\quad\quad\quad\quad\quad\quad\quad  Application: Pretrained model selection     & $Trf(\mathcal{D}_s, \mathcal{D}_t,\mathcal{D}_t)$ \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad Application: Task embedding, dataset similarity, domain selection       \\
Instance-level & $Trf((f_s [,\mathcal{D}_s]), \mathcal{D}_t , (x_t,y_t))$ \quad\quad\quad\quad\quad\quad\quad\quad Application: Instance-adaptive transfer learning & $Trf(\mathcal{D}_s,[\mathcal{D}_t], (x_t,y_t)) $ \quad\quad\quad\quad\quad\quad\quad\quad  Application: OOD detection, active sample selection in domain adaptation \\ \bottomrule
\end{tabular}
\caption{A taxonomy of transferability based on the knowledge modality being transferred (column) and granuality of transferability evaluation (rows).}
\end{table}

Having categorized “what to transfer” into the above three categories, we can discuss transferability measurement for each of these categories. We will consider the following question: what do we mean by transferability of a model, data and etc, and how to go about measuring them from data. 

\begin{itemize}
\item {\bf Model transferability $Trf( ( f_s [,\mathcal{D}_s]), \mathcal{D}_t, \cdot)$. }  Measuring the transferability in model transfer depends on estimating the optimal performance of the target model given source model information. Based on the distribution assumptions of the source and target tasks in model transfer, model transferability may further be categorized as cross-domain, cross-task etc. Meanwhile, whether the source data is used in addition to source model would lead to {\bf source-dependent} and {\bf source-free} transferability metrics.

% {\bf Prompt and adversarial example transferability. }$Trf((\delta_s [,\mathcal{D}_s ] ),(f_t,\mathcal{D}_t), \cdot)$ Depending on whether the same prompt is being transferred acrcoss different models, or whether a prompt is adapted to cross different tasks, they are referred as  {\bf cross-model} and {\bf cross-task} prompt transfer, resepectively \cite{su2021transferability}. Given a source perturbation $\delta_s$, potentially generated based on a source dataset $\mathcal{D}_s$, and a target model $f_t$ operating on a target dataset $\mathcal{D}_t$, transferability of the perturbation or prompt can be measured through its effectiveness in achieving a specific objective (e.g., fooling the target model in adversarial examples, or successfully transferring prompts across models or tasks).

% For both prompt and adversarial example transferability, the goal is to optimize a transferability function based on whether the transfer is across models or tasks, denoted as \textbf{cross-model} and \textbf{cross-task} transferability, respectively.

% \noindent{\bf Adversarial Example Transferability.}
% Adversarial example transferability focuses on the transfer of adversarial perturbations $\delta_s$ crafted for a source model $f_s$ and how well they can be transferred to fool a target model $f_t$ on a different task or domain.

% The transferability of adversarial examples can be expressed as:
% \[
% Trf((\delta_s [,\mathcal{D}_s]), (f_t, \mathcal{D}_t), \cdot) := \max_{\delta_s} \mathbb{E}_{(x,y) \sim D_t} \mathcal{L}(f_t(x + \delta_s), y),
% \]
% where $\mathcal{L}(\cdot)$ is the loss function, $(x, y) \sim D_t$ are samples from the target dataset, and $\delta_s$ represents the adversarial perturbation, typically generated using a surrogate model or data from the source domain. The goal is to maximize the loss for the target model $f_t$, thereby ensuring that $\delta_s$ successfully transfers.

\noindent{\bf Prompt Transferability.}
As a special case of model transferability, prompt transferability measures the effectiveness of transferring a prompt $\delta_s$ generated for a source model $f_s$ or dataset $\mathcal{D}_s$, to a target model $f_t$ or task $\mathcal{D}_t$. Prompt transferability can be formulated as:
\[
Trf((\delta_s [,\mathcal{D}_s]), (f_t, \mathcal{D}_t), \cdot) := \min_{\delta_s} \mathbb{E}_{(x,y) \sim D_t} \mathcal{L}(f_t(\delta_s(x)), y),
\]
where the objective is to minimize the loss $\ell(\cdot)$, ensuring that the prompt successfully transfers to the target model or task.

\item {\bf Dataset transferability $Trf(\mathcal{D}_s, \mathcal{D}_t,\cdot)$. }  From a transferability measurement perspective, thanks to the generalization analysis of domain adaptation, we can approximte transferability by statistical divergence between source and target distributions. Therefore dataset transfer  essentially estimates the relationship between two tasks irrespective to the source model choice. 

{\bf Relationship with model transferability. } It can be argued if the source model (such as a deep network)   has sufficient capacity and is well-trained, model architecture makes no significant difference in its transfer performance as long as the source and target datasets are fixed. Therefore, one could potentially define dataset transferability via a model transferability with an optimal source model. 
\[Trf(\mathcal{D}_s,\mathcal{D}_t,\cdot):=\max_{f_s} Trf((f_s,\mathcal{D}_s),\mathcal{D}_t,\cdot)\]

% \item{\bf Prompt and adversarial example transferability. }$Trf((\delta_s [,\mathcal{D}_s ] ),(f_t,\mathcal{D}_t), \cdot)$ Depending on whether the same prompt is being transferred acrcoss different models, or whether a prompt is adapted to cross different tasks, they are referred as  {\bf cross-model} and {\bf cross-task} prompt transfer, resepectively \cite{su2021transferability}. Transferability estimation for these variants…. (to be completed after more paper review)

% \noindent{\bf General Formulation.}
% Given a source perturbation $\delta_s$, potentially generated based on a source dataset $\mathcal{D}_s$, and a target model $f_t$ operating on a target dataset $\mathcal{D}_t$, transferability of the perturbation or prompt can be measured through its effectiveness in achieving a specific objective (e.g., fooling the target model in adversarial examples, or successfully transferring prompts across models or tasks).

% For both prompt and adversarial example transferability, the goal is to optimize a transferability function based on whether the transfer is across models or tasks, denoted as \textbf{cross-model} and \textbf{cross-task} transferability, respectively.

% \noindent{\bf Adversarial Example Transferability.}
% Adversarial example transferability focuses on the transfer of adversarial perturbations $\delta_s$ crafted for a source model $f_s$ and how well they can be transferred to fool a target model $f_t$ on a different task or domain.

% The transferability of adversarial examples can be expressed as:
% \[
% Trf((\delta_s [,\mathcal{D}_s]), (f_t, \mathcal{D}_t), \cdot) := \max_{\delta_s} \mathbb{E}_{(x,y) \sim D_t} \mathcal{L}(f_t(x + \delta_s), y),
% \]
% where $\mathcal{L}(\cdot)$ is the loss function, $(x, y) \sim D_t$ are samples from the target dataset, and $\delta_s$ represents the adversarial perturbation, typically generated using a surrogate model or data from the source domain. The goal is to maximize the loss for the target model $f_t$, thereby ensuring that $\delta_s$ successfully transfers.

% \noindent{\bf Prompt Transferability.}
% Similarly, prompt transferability measures the effectiveness of transferring a prompt $\delta_s$ generated for a source model $f_s$ or dataset $\mathcal{D}_s$, to a target model $f_t$ or task $\mathcal{D}_t$.

% Prompt transferability can be formulated as:
% \[
% Trf((\delta_s [,\mathcal{D}_s]), (f_t, \mathcal{D}_t), \cdot) := \min_{\delta_s} \mathbb{E}_{(x,y) \sim D_t} \mathcal{L}(f_t(\delta_s(x)), y),
% \]
% where the objective is to minimize the loss $\ell(\cdot)$, ensuring that the prompt successfully transfers to the target model or task.

\end{itemize}

### Granuality of Transferability Evaluation
\begin{itemize}
\item {\bf Task level $Trf(\cdot, \cdot,\mathcal{D}_t)$. } Also known as domain level or population level. Computes transferability over multiple target %training (target?)
samples. This is the most common form of transferability estimation. 

{\bf Region level  $Trf(\cdot, \cdot,  R)$. }  As a special case of task level transferability, which uses a subset of target samples for evaluation, in tasks with high dimensional output, such as 2D and 3D semantic segmentation, the transferability of source knowledge modality  can vary by region or semantic class. Let  $R \subset \mathcal{D}_t$  represent the target samples belonging to the same region or semantic class. If we treat the model  prediction over $R$  as an individual task,   the transfer performance on   subset  $R$   can be seen as a special kind of task-level transferability.

\item {\bf Instance level $Trf(\cdot, \cdot,\{x_t,y_t\})$. } Computes transferability with respect to a single sample. Methods based on population statistics (e.g. covariance) of the target data, are no longer applicable. On the other hand, models based on sample hypothesis testing. 
% TODO: 放到instance level里面
% TOOO: 还是放到了task level里面
\end{itemize}


### When is Transferability Computed

Transferability metrics are often designed to be applied at different stages of  target model training. With different design goals, the available information and the computation requirement vary, each bringing unique advantage and shortcomings. 

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\linewidth]{pictures/timeline.png}
    \caption{Three stages (pre-hoc, online, post-hoc) of when transferability is measured.}
    \label{fig:enter-label}
\end{figure}

{\bf Before training (pre-hoc). }  Often used as a filter to select transferable task or model prior to training, it needs to be efficient and easy to compute. Therefore statistical estimation of transferability or data-driven approach to predict transferability from available task information are preferred.

{\bf During training (online). }  Allows the utilization of gradient information or weight information during training to compute transferability. Due to the need for training information, such methods are not practicle for offline model selection. However, it is useful in dynamic learning algorithms, offering adaptive control of the transfer process.   

{\bf After training (post-hoc). } Utilizes the final training loss or accuracy to measure transferability empirically. Its obvious drawback is the computation efficiency.  As a post-hoc method, it is mostly used for analyzing and storing transferability of certain types of tasks or models.  Nevertheless, one could still use it with a small validation dataset to perform approximated model selection in a timely manner.  




 ### Beyond Pairwise Transferability
 Traditionally transferability is defined between a single source task and a single target task. Overtime, researchers have extended these pairwise transferability to involve multiple tasks. e.g. In { multi-source transferability}, the goal is to evaluate how an ensemble of source tasks perform on the target task. 
 Besides computing pairwise and multi-source transferability, researchers have approached transferability estimation from a {\bf metric learning} perspective. i.e. Using pre-computed pairwise transferability among a large zoo of tasks, each containing  model and dataset attributes, find a task embedding such that the embedding distance  approximates  the pairwise task transferability. \cite{zhang2021quantifying} provides a method to compute the transferability among multiple domains by considering the largest domain gap.



