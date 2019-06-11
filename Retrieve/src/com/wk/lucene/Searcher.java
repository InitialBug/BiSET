package com.wk.lucene;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import com.wk.lucene.Constants;

public class Searcher {

    public static void search(String indexDir, String queryPath,String fileName, int query_num) throws Exception {
    	
    	FileWriter writer=new FileWriter(fileName,true);
    	
        Directory dir = FSDirectory.open(Paths.get(indexDir));
        IndexReader reader = DirectoryReader.open(dir);
        IndexSearcher is = new IndexSearcher(reader);
        Analyzer analyzer = new EnglishAnalyzer();
   
        QueryParser parser = new QueryParser("contents", analyzer);
        int num=0;
        String line="";
        BufferedReader in=new BufferedReader(new FileReader(queryPath));
        while(true){
        	System.out.println(num);
        	num++;
            line=in.readLine();
            if(line==null)
            	break;
            
            line=line.replaceAll("[^a-z<>#\\s]" , "");       

        	System.out.println(line);
            Query query = parser.parse(line);
 
       
            TopDocs hits = is.search(query, query_num);

 
            for (ScoreDoc scoreDoc : hits.scoreDocs) {
                Document doc = is.doc(scoreDoc.doc);
                writer.write(doc.get("fileName")+' ');
                 }
            writer.write('\n');

        }
        reader.close();
        writer.close();

    }

    public static void main(String[] args) throws IOException {
    	
    	//the directory of index
        String indexDir = Constants.indexDir;
        //put the queries in one file, each line contains one query 
        String queryPath = Constants.queryPath;
        //the file to save the results, each line is the index of indexed file
    	String results=Constants.results;
    	// query number
    	int query_num=Constants.query_num;

        try {
            search(indexDir, queryPath,results,query_num);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}