(defproject ml "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.clojure/clojurescript "1.9.660"]]
  :profiles {:dev {:dependencies [[org.clojure/data.csv "0.1.4"]
                                  [org.clojure/test.check "0.9.0"]]
                   :resource-paths ["test-resources"]}}

  :source-paths ["src/cljc"])
