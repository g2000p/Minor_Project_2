#include "olcConsoleGameEngine.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include <string>
#include <cmath>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

vector<string> split(string str, char c) {
	vector<string> array;
	string element = "";

	for (unsigned i = 0; i < str.length(); i++) {
		if (str[i] != c)
			element += str[i];
		else if (str[i] == c && element != "") {
			array.push_back(element);
			element = "";
		}
	} if (element != "")
		array.push_back(element);

	return array;
}

class File {

private:
	int max_iterations, data_size;
	vector<unsigned> layout;
	vector< vector<double> > inputs;
	vector< vector<double> > targets;

public:
	vector<double> getinputs(const int index) const { return inputs[index]; };
	vector<double> gettargets(const int index) const { return targets[index]; };
	vector<unsigned> getlayout() const { return layout; };
	int getmaxiterations() const { return max_iterations; };
	int getdatasize() const { return data_size; };

	File(const char* filepath) {
		string line;
		vector<string> part;
		ifstream file(filepath);

		if (file.is_open()) {
			int index = 0;
			while (getline(file, line)) {
				if (index == 0)
					max_iterations = atoi(line.c_str());
				else if (index == 1) {
					part = split(line, ' ');
					for (unsigned p = 0; p < part.size(); p++)
						layout.push_back(atoi(part[p].c_str()));
				}
				else if (index % 2 == 0) {
					vector<double> i;
					part = split(line, ' ');
					for (unsigned p = 0; p < part.size(); p++)
						i.push_back(atof(part[p].c_str()));
					inputs.push_back(i);
				}
				else {
					vector<double> t;
					part = split(line, ' ');
					for (unsigned p = 0; p < part.size(); p++)
						t.push_back(atof(part[p].c_str()));
					targets.push_back(t);
				}
				index++;
			}
		}
		data_size = inputs.size();
		file.close();
	}

};
