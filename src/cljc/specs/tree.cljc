(ns specs.tree
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [ml.tree :as tree]))

(s/def ::pos-int (s/and int? #(> % 0)))

(s/def ::vector-of-same-type-data (s/or :number-coll (s/coll-of number? :kind vector?)
                                        :string-coll (s/coll-of string? :kind vector?)
                                        :keyword-coll (s/coll-of keyword? :kind vector?)))

;; train-decision-tree specs

(s/def ::tree-input-data-values (s/and #(->> % meta :type #{:categorical :numerical})
                                       ::vector-of-same-type-data))

(s/def ::tree-input-data (s/with-gen (s/or :empty {}
                                           :not-empty (s/and (s/map-of keyword? ::tree-input-data-values)
                                                             #(if (< 1 (count %))
                                                                (apply = (map count (vals %)))
                                                                true)
                                                             #(let [possible-ys (keys %)
                                                                    y (-> % meta :y)]
                                                                (and (keyword? y)
                                                                     (some (fn [possible-y]
                                                                             (= possible-y y))
                                                                           possible-ys)))))
                                     #(let [n (rand-int 50)]
                                        (gen/fmap (fn [data-map]
                                                    (if (empty? data-map)
                                                      {}
                                                      (let [input-data-meta {:y (key (first data-map))}]
                                                        (with-meta (into {}
                                                                         (map (fn [[k v]]
                                                                                (let [input-data-values-meta {:type (if (number? (first v))
                                                                                                                      :numerical
                                                                                                                      :categorical)}]
                                                                                  [k (with-meta v input-data-values-meta)]))
                                                                              data-map))
                                                                   input-data-meta))))
                                                  (s/gen (s/map-of keyword? (s/or :number-coll (s/coll-of number? :kind vector? :count n)
                                                                                  :string-coll (s/coll-of string? :kind vector? :count n)
                                                                                  :keyword-coll (s/coll-of keyword? :kind vector? :count n))))))))

(s/def ::stop-criterion ::pos-int)
(s/def ::binary-tree? boolean?)
(s/def ::cost-function (s/nilable #{:information-gain :gini}))

(s/def ::tree-input-options (s/keys :req-un [::stop-criterion]
                                    :opt-un [::binary-tree? ::cost-function]))

(s/def ::leaf any?)
(s/def ::test fn?)
(s/def ::cost number?)
(s/def ::best-split-value number?)
(s/def ::branches (s/or :binary-tree #(= % [true false])
                        :not-binary-tree (s/coll-of any?)))
(s/def ::data-key keyword?)
(s/def ::split-point any?)
(s/def ::attribute-data (s/keys :req-un [::split-point ::branches ::data-key]
                                :req-opt [::cost ::best-split-value]))
(s/def ::n-data-points ::pos-int)

(s/def ::tree (s/or
                :leaf (s/keys :req-un [::leaf])
                :node (s/and (s/keys :req-un [::test ::attribute-data ::n-data-points])
                             (fn [node]
                               (let [branches-map (dissoc node :test :attribute-data :n-data-points)]
                                 (and (>= 2 (count branches-map))
                                      (not (some false?
                                                 (map #(not= :clojure.spec.alpha/invalid (s/conform ::tree %))
                                                      (vals branches-map))))))))))

(s/fdef tree/train-decision-tree
  :args (s/cat :data ::tree-input-data :options ::tree-input-options)
  :ret ::tree
  :fn #(if (-> % :args :options :binary-tree?)
         (tree/test-tree (fn [node]
                           (or (= (keys node) [:test :attribute-data :n-data-points true false])
                               (contains? node :leaf)))
                         (:ret %)
                         {})
         true))

;; best-attribute specs

(s/def ::vector-of-data (s/and #(->> % meta :type #{:categorical :numerical})
                               ::vector-of-same-type-data))

(s/def ::explained-data ::vector-of-data)
(s/def ::explaining-data-sets (s/map-of keyword? ::vector-of-data))

(s/fdef tree/best-attribute
  :args (s/cat :explained-data ::explained-data :explaining-data-sets ::explaining-data-sets :options ::tree-input-options)
  :ret ::attribute-data)