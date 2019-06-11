package com.wk.lucene;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import com.wk.lucene.Constants;

/*build index */
public class Indexer {
        private IndexWriter writer;

    /**
     * @param indexDir
     * @throws IOException
     */
    public Indexer(String indexDir) throws IOException {
        //get the directory of index  
        Directory directory = FSDirectory.open(Paths.get(indexDir));
        // use the EnglishAnalyzer to tokenize
        Analyzer analyzer = new EnglishAnalyzer();
        //save the config
        IndexWriterConfig iwConfig = new IndexWriterConfig(analyzer);

        writer = new IndexWriter(directory, iwConfig);
    }

    /**
     * close index
     * 
     * @throws Exception
     * @return the number of indexed documents
     */
    public void close() throws IOException {
        writer.close();
    }

    public int index(String dataDir) throws Exception {
        File[] files = new File(dataDir).listFiles();
        for (File file : files) {
            
            indexFile(file);
        }
        return writer.numDocs();

    }

    /**
     * index target file
     * 
     * @param file
     */
    private void indexFile(File f) throws Exception {
        System.out.println("index file：" + f.getCanonicalPath());
        Document doc = getDocument(f);
        writer.addDocument(doc);
    }


    private Document getDocument(File f) throws Exception {
        Document doc = new Document();
        doc.add(new TextField("contents", new FileReader(f)));
        doc.add(new TextField("fileName", f.getName(), Field.Store.YES));
        doc.add(new TextField("fullPath", f.getCanonicalPath(), Field.Store.YES));
        return doc;
    }

    public static void main(String[] args) {
    	try {
			TimeUnit.MINUTES.sleep(10);
		} catch (InterruptedException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
        //directory to save the index
        String indexDir = Constants.indexDir;
        
        //the target directory to be indexed, put the data in separate files, each file contains one data   
        String dataDir = Constants.dataDir;
        Indexer indexer = null;
        int numIndexed = 0;
        long start = System.currentTimeMillis();
        try {
            indexer = new Indexer(indexDir);
            numIndexed = indexer.index(dataDir);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } finally {
            try {
                indexer.close();
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
        long end = System.currentTimeMillis();
        System.out.println("index：" + numIndexed + " files cost" + (end - start) + " ms");
    }

}