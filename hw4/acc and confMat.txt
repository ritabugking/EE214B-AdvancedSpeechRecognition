function [testacc, cm] = calcCM(labels, outputs, testInd)
% Usage: [testacc, cm] = calcCM(labels, outputs, testInd)
% labels are true values and outputs are predicted values
% testInd denotes test indices of the data
% These variables have the same name in hw4.m
% testacc is overall test accuracy
% cm is the confusion matrix

[~, labels] = max(labels, [], 1);
[~, outputs] = max(outputs, [], 1);
cm = confusionmat(labels(testInd), outputs(testInd));
testacc = trace(cm)/sum(cm(:));

end
