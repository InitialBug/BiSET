package com.wk.lucene;

public class Constants {
    //directory to save the index
	public static final String indexDir = "/home/k/Data/index_article";
    
    //the target directory to be indexed, put the data in separate files, each file contains one data   
	public static final String dataDir = "/home/k/Data/train_article";
	
	//put the queries in one file, each line contains one query 
	public static final String queryPath = "/home/k/Data/train.article.txt";
	
    //the file to save the results, each line is the index of indexed file
	public static final String results="/home/k/Data/train.template.index";
	
	// query number
	public static final int query_num=30;

}
