#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
using edge = pair<pair<int,int>,int>;




pair<vector<edge>,vector<int>> parser(){
    string garbage;
    int n;
    int nb_node,nb_edges;
    cin>>garbage;
    cin>> garbage;
    cin>> garbage;
    cin>> nb_node;
    cin>> garbage;
    cin>> nb_edges;
    vector<edge> edges;
    for(int i =0; i<nb_edges; i++){
        edge e;
        int a,b,w;
        char c;
        cin>>c>>a>>b>>w;
        e = {{a,b},w};
        edges.push_back(e);
    }
    int nb_terminal;
    cin>> garbage;
    cin>> garbage;
    cin>> garbage;
    cin>> garbage;
    cin>> nb_terminal;
    vector<int> terminals;
    for(int i =0; i<nb_terminal; i++){
        int t;
        char c;
        cin>>c>>t;
        terminals.push_back(t);
    }
    pair<vector<edge>,vector<int>> a = {edges,terminals};
    return(a);
}




int main()
{
    pair<vector<edge>,vector<int>> a = parser();
    vector<edge> edges = a.first;
    vector<int> terminals = a.second;
    cout<<terminals.size();
    return 0;
}
