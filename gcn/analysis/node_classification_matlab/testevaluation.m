% note that Y must be a sparse matrix


Y = sign(sprand(5, 2, 0.5));
pred = Y;


evaluate(pred, Y)

pred = rand(5, 2);
evaluate(pred, Y)
