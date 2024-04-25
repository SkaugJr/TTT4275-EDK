function predicted_labels = classify_nn(train_data, train_labels, test_data)
    num_test = size(test_data, 1);
    num_train = size(train_data, 1);
    predicted_labels = zeros(num_test, 1);
    
    for i = 1:num_test
        distances = zeros(num_train, 1);
        for j = 1:num_train
            distances(j) = sqrt(sum((test_data(i, :) - train_data(j, :)).^2));
        end
        [~, min_index] = min(distances);
        predicted_labels(i) = train_labels(min_index);
    end
end

