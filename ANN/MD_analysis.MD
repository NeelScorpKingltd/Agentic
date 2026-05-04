MODEL Analysis

	LOSS/EPOCH					
No. of Hidden Layers	1	2	3	4	5	Accuracy
1	        0.4282	0.2316	0.1845	0.1626	0.1434	97.01
2	        0.4999	0.2631	0.2176	0.1891	0.1758	96.58
3	        0.6053	0.3098	0.2557	0.255	0.2033	96.23
4	        0.422	0.2298	0.181	0.1576	0.1431	96.58

The ANN model demonstrates strong performance across all configurations, achieving accuracies between 96.23% and 97.01% on the MNIST test set. Notably, the single hidden layer architecture performs best, achieving the highest accuracy of 97.01% with the lowest final epoch loss of 0.1434, suggesting that a simpler model is more effective for this task. Adding more hidden layers (2-4) does not improve performance and instead increases the final loss values, with the 3-layer model showing the poorest results (96.23% accuracy, 0.2033 loss). The loss curves demonstrate consistent convergence across epochs in the single-layer model, while multi-layer models exhibit higher training losses despite adequate learning. This indicates that model complexity beyond a single hidden layer is unnecessary for this problem, and that a simpler architecture provides better generalization while reducing computational overhead and training time.

    LOSS/EPOCH					
No. of nodes 	1	2	3	4	5	Accuracy
784-64	    0.5035	0.3001	0.2545	0.2289	0.2136	95.58
784-128	    0.4388	0.241	0.1932	0.1702	0.1527	96.77
784-256	    0.3739	0.1913	0.152	0.1285	0.1167	96.36


The ANN model achieves optimal performance with 128 hidden nodes, reaching the highest test accuracy of 96.77% with well-controlled loss convergence (0.1527 at epoch 5). The 64-node configuration significantly underperforms with 95.58% accuracy, indicating insufficient model capacity to learn the data complexity. Conversely, increasing to 256 nodes lowers the final training loss to 0.1167 (the lowest across configurations) but paradoxically reduces test accuracy to 96.36%, suggesting incipient overfitting. This classic bias-variance tradeoff demonstrates that 128 nodes provides the sweet spot—sufficient representational capacity to capture MNIST's complexity while maintaining strong generalization. The findings confirm that 128 neurons is the optimal hidden layer size for this task, balancing model efficiency with predictive performance without the diminishing returns of excessive parameterization.

    LOSS/EPOCH					
Batch Size	1	2	3	4	5	Accuracy
32	    0.3901	0.2199	0.1798	0.1587	0.1448	96.79
64	    0.4388	0.241	0.1932	0.1702	0.1527	96.77
128	    0.4612	0.2469	0.1932	0.1632	0.1432	96.6
256 	0.5784	0.3039	0.2412	0.2001	0.178	95.73


The best batch size from results is 32, because it produces the highest test accuracy at 96.79% and the lowest final training loss at 0.1448. Compared to larger batches, it converges faster and maintains better generalization, while batch sizes of 128 and 256 show weaker accuracy and higher loss. Batch size 64 is close, but 32 still edges it out on both accuracy and loss. So, for this ANN and MNIST setup, batch size 32 is the best choice.


Based on above analysis, below mentioned config gives the strongest balance of generalization and training behavior: the single-layer model reached the highest accuracy, 128 nodes provided the best accuracy/loss tradeoff, and batch size 32 delivered the best test accuracy with the lowest final loss.

he best ANN setup is:
Architecture: 1 hidden layer
Hidden nodes: 128
Batch size: 32
Dropout: 0.2
Learning rate: 0.001
Epochs: 5