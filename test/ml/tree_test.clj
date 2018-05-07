(ns ml.tree-test
  (:require [clojure.test :refer :all]
            [clojure.data.csv :as csv]
            [clojure.string :as clj-str]
            [clojure.java.io :as io]
            [ml.tree :refer :all]))

(defn read-csv [path]
  (with-open [reader (io/reader path)]
    (doall
      (csv/read-csv reader))))

(defn format-text
  [txt]
  (-> txt
      (clj-str/replace #"[\.\s]" "-")
      (clj-str/lower-case)))

(defn split-data
  [txt]
  (clj-str/split txt #" "))

(def iris-tree
  {:test "test-fn"
   :attribute-data {:cost 1/3 :data-key :petal-length :split-point "2.500" :branches [true false]}
   :n-data-points 75
   true {:leaf "setosa"}
   false {:test "test-fn"
          :attribute-data {:cost 2/27 :data-key :petal-length :split-point "4.800" :branches [true false]}
          :n-data-points 50
          true {:leaf "versicolor"}
          false {:test "test-fn"
                 :attribute-data {:cost 4/45 :data-key :petal-width :split-point "1.750" :branches [true false]}
                 :n-data-points 27
                 true {:test "test-fn"
                       :attribute-data {:cost 4/15 :data-key :petal-width :split-point "1.550" :branches [true false]}
                       :n-data-points 5
                       true {:leaf "virginica"}
                       false {:leaf "versicolor"}}
                 false {:leaf "virginica"}}}})

(def wrongly-predicted-iris-data-points
  [{:sepal-length (float 6.9) :sepal-width (float 3.1) :petal-length (float 4.9) :petal-width (float 1.5) :species "versicolor"}
   {:sepal-length (float 5.9) :sepal-width (float 3.2) :petal-length (float 4.8) :petal-width (float 1.8) :species "versicolor"}
   {:sepal-length (float 6.3) :sepal-width (float 2.5) :petal-length (float 4.9) :petal-width (float 1.5) :species "versicolor"}
   {:sepal-length (float 6.8) :sepal-width (float 2.8) :petal-length (float 4.8) :petal-width (float 1.4) :species "versicolor"}
   {:sepal-length (float 4.9) :sepal-width (float 2.5) :petal-length (float 4.5) :petal-width (float 1.7) :species "virginica"}])

(deftest test-decision-tree-training
  (let [iris-data (read-csv "test-resources/iris.csv")
        abalone-data (read-csv "test-resources/abalone.csv")
        parse-data-fn (fn [data]
                        (apply map
                               (fn [heading & args]
                                 (let [first-half (keep identity
                                                        (map-indexed (fn [i v]
                                                                       (when (odd? i)
                                                                         v))
                                                                     args))
                                       second-half (keep identity
                                                         (map-indexed (fn [i v]
                                                                        (when-not (odd? i)
                                                                          v))
                                                                      args))
                                       parse-data #(try (Float. %) (catch Exception e (when-not (empty? %) %)))]
                                   {:test {(keyword (format-text heading)) (mapv parse-data first-half)}
                                    :train {(keyword (format-text heading)) (mapv parse-data second-half)}}))
                               data))
        parsed-iris-data (parse-data-fn (map #(split-data (first %))
                                             (butlast iris-data)))
        parsed-abalone-data (parse-data-fn abalone-data)
        iris-data-ready-for-training (with-meta (apply merge
                                                       (map #(let [[k v] (first (:test %))]
                                                               {k (with-meta v (if (= k :species)
                                                                                 {:type :categorical}
                                                                                 {:type :numerical}))})
                                                            parsed-iris-data))
                                                {:y :species})
        #_#_abalone-data-ready-for-training (with-meta (apply merge
                                                          (map #(let [[k v] (first (:test %))]
                                                                  {k (with-meta v (if (= k :sex)
                                                                                    {:type :categorical}
                                                                                    {:type :numerical}))})
                                                               parsed-abalone-data))
                                                   {:y :rings})

        trained-iris-tree (train-decision-tree iris-data-ready-for-training {:stop-criterion 3 :cost-function :gini :binary-tree? true})
        #_#_trained-abalone-tree (train-decision-tree abalone-data-ready-for-training {:stop-criterion 3 :cost-function :gini :binary-tree? true})
        iris-test-data (map-vecs->vec-maps (apply merge (map :train parsed-iris-data)))]
    (is (= iris-tree (traverse-tree trained-iris-tree {:attribute-data #(update % :split-point (fn [value]
                                                                                    (format "%.3f" value)))
                                          :test (fn [_] "test-fn")})))
    (is wrongly-predicted-iris-data-points (keep #(when (not= (apply-tree trained-iris-tree %) (:species %)) %)
                                                 iris-test-data))))
