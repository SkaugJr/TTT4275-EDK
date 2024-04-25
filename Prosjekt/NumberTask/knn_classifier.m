function predicted_labels = knn_classifier(train_data, train_labels, test_data, K)
    num_test = size(test_data, 1);
    predicted_labels = zeros(num_test, 1);
   
    distances = pdist2(test_data, train_data);
    
    [~, indices] = mink(distances, K, 2);
    
    for i = 1:num_test
        nearest_labels = train_labels(indices(i, :));
        predicted_labels(i) = mode(nearest_labels);
    end
end

