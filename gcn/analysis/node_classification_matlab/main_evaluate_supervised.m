tscores = load('gcn_blog_tscores_180.mat');
tscores = single(tscores.arr);
tids = load('gcn_blog_tids.mat');
tids = tids.arr+1;

group = load('blog_labels.mat');
group = group.arr;

load blogcatalog;

[perf, pred] = evaluate(tscores, group(tids, :));

macrof1 = perf.macro_F1
microf1 = perf.micro_F1
