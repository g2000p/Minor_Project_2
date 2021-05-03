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

//connection
struct Connection {
	double weight;
	double dweight;
};



class Neuron {

public:
	static double learningrate;
	static double alpha;
	
	static double activate(double value)
	{
		return 1 / (1 + exp(-value));
	}
	
	static double activateD(double value)
	{
		return activate(value) * (1 - activate(value));
	}
	static double random(void) {
		return rand() / double(RAND_MAX);
	}
	double sumDOW(const Layer& nextlayer)
	{
		double sum = 0.0;

		for (unsigned n = 0; n < nextlayer.size() - 1; n++)
			sum += outweight[n].weight * nextlayer[n].gradient;

		return sum;
	}

	double output;
	vector<Connection> outweight;
	unsigned index;
	double gradient;

	Neuron(unsigned outamt, unsigned index)
	{
		this->index = index;
		outweight.reserve(outamt);

		for (unsigned i = 0; i < outamt; i++) {
			outweight.push_back(Connection());
			outweight.back().weight = random();
		}
	}

	void setoutput(double value)
	{
		output = value;
	}

	double getout(void) const
	{
		return output;
	}

	vector<Connection> getoutweight() const
	{
		return outweight;
	}

	void feedforward(const Layer& prevlayer)
	{
		double sum = 0.0;

		for (unsigned n = 0; n < prevlayer.size(); n++)
			sum += prevlayer[n].getout() * prevlayer[n].outweight[index].weight;

		output = Neuron::activate(sum);
	}

	void outgradient(double target)
	{
		double delta = target - output;
		gradient = delta * Neuron::activateD(output);
	}

	void hidgradient(const Layer& nextlayer)
	{
		double dow = sumDOW(nextlayer);
		gradient = dow * Neuron::activateD(output);
	}

	void updateweight(Layer& prevlayer)
	{

		for (unsigned n = 0; n < prevlayer.size(); n++)
		{
			double olddweight = prevlayer[n].outweight[index].dweight;

			double newdweight = learningrate * prevlayer[n].getout() * gradient + alpha * olddweight;

			prevlayer[n].outweight[index].dweight = newdweight;
			prevlayer[n].outweight[index].weight += newdweight;
		}
	}

};

double Neuron::learningrate = 0.2;
double Neuron::alpha = 0.5;

class Network {

private:
	vector<Layer> layers;
	double error;
	double avgerror;
	static double smoothfact;

public:
	Network(const vector<unsigned>& layout)
	{
		layers.reserve(layout.size());
		error = 0;
		avgerror = 0;
		for (unsigned l = 0; l < layout.size(); l++) {
			layers.push_back(Layer());
			unsigned outamt = l == layout.size() - 1 ? 0 : layout[l + 1];

			layers.back().reserve(layout[l]);
			for (unsigned n = 0; n <= layout[l]; n++)
				layers.back().push_back(Neuron(outamt, n));

			layers.back().back().setoutput(1);
		}
	}
	void feedforward(const vector<double>& inpvals)
	{
		for (unsigned i = 0; i < inpvals.size(); i++) {
			layers[0][i].setoutput(inpvals[i]);
		}

		for (unsigned lnum = 1; lnum < layers.size(); lnum++) {
			for (unsigned n = 0; n < layers[lnum].size() - 1; n++)
				layers[lnum][n].feedforward(layers[lnum - 1]);
		}
	}
	void backprop(const vector<double>& targets)
	{
		Layer& outlayer = layers.back();
		error = 0.0;

		for (unsigned n = 0; n < outlayer.size() - 1; n++) {
			double delta = targets[n] - outlayer[n].getout();
			error += delta * delta;
		}
		error /= outlayer.size() - 1;
		error = sqrt(error);

		avgerror = (avgerror * smoothfact + error) / (smoothfact + 1.0);

		for (unsigned n = 0; n < outlayer.size() - 1; n++) {
			outlayer[n].outgradient(targets[n]);
		}

		for (unsigned lnum = layers.size() - 2; lnum > 0; --lnum) {
			for (unsigned n = 0; n < layers[lnum].size(); n++) {
				layers[lnum][n].hidgradient(layers[lnum + 1]);
			}
		}

		for (unsigned lnum = layers.size() - 1; lnum > 0; lnum--) {
			for (unsigned n = 0; n < layers[lnum].size() - 1; n++) {
				layers[lnum][n].updateweight(layers[lnum - 1]);
			}
		}
	}
	void getres(vector<double>& results)
	{
		results.clear();
		results.reserve(layers.back().size());

		for (unsigned n = 0; n < layers.back().size() - 1; n++)
		{
			results.push_back(layers.back()[n].getout());
		}
	}
	double recentavgerror(void) const { return avgerror; }
	vector<Layer> getLayers() const { return layers; }

};

