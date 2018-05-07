(ns ml.core
  (:require [ml.helpers :as helpers]))

(defn ready-function
  [w-first-layer w-second-layer]
  (fn [x]
    (let [x (conj x 1)
          hidden-units (map #(helpers/basis-function :neural-network {:x x :w % :activation-fn :tanh})
                            w-first-layer)]
      (helpers/linear-model :regression {:w (first w-second-layer) :basis-fns hidden-units :activation-fn :identity}))))

(defmulti neural-network
          (fn [_ {method :method}]
            method))

(defmethod neural-network :feed-forward
  [observations
   {:keys [max-iterations learning-rate gradient-limit] :as options}]
  (let [vector-valued-observation? (vector? (:x (first observations)))
        max-iterations (or max-iterations 100)
        M 4
        learning-rate (or learning-rate 0.4)
        gradient-limit (or gradient-limit 0.005)]
    (loop [weights []]
      (when-not (empty? weights)
        (let [w-first-layer (first weights)
              w-second-layer (second weights)
              ready-function (ready-function w-first-layer w-second-layer)
              y-diff (mapv #(- (first (:t %)) (ready-function (:x %))) observations)]
          (println "READY ----> " (pr-str weights))
          (println "SUM -------> " (pr-str (apply + y-diff)))
          (println "--------------------------")))
      (recur (reduce (fn [layer-weights {:keys [x t] :as observation}]
                       (let [x (conj x 1)]
                         (loop [i 1
                                w-first-layer (if (empty? layer-weights)
                                                (into [] (repeat M
                                                                 (-> (count x) (- 1) (take (iterate inc 1)) vec (conj 1))))
                                                (first layer-weights))
                                w-second-layer (if (empty? layer-weights)
                                                 (into [] (repeat (count t)
                                                                  (-> M (- 1) (take (iterate inc 1)) vec (conj 1))))
                                                 (second layer-weights))]
                           (if (= i 2)
                             [w-first-layer w-second-layer]
                             (let [hidden-units (map #(helpers/basis-function :neural-network {:x x :w % :activation-fn :tanh})
                                                     w-first-layer)
                                   y (map #(helpers/linear-model :regression {:w % :basis-fns hidden-units :activation-fn :identity})
                                          w-second-layer)
                                   output-errors (map #(- %1 %2) y t)
                                   hidden-unit-errors (map-indexed (fn [index hidden-unit]
                                                                     (helpers/error-backpropagate :tanh {:hidden-unit hidden-unit
                                                                                                         :output-errors output-errors
                                                                                                         :w (helpers/column w-second-layer index)}))
                                                                   hidden-units)
                                   first-layer-error-derivatives (mapv (fn [error]
                                                                         (mapv #(* error %)
                                                                               x))
                                                                       hidden-unit-errors)
                                   second-layer-error-derivatives (mapv (fn [error]
                                                                          (mapv #(* error %)
                                                                                hidden-units))
                                                                        output-errors)]
                               (recur (inc i)
                                      (helpers/M-minus w-first-layer (helpers/M-element-multiply learning-rate first-layer-error-derivatives))
                                      (helpers/M-minus w-second-layer (helpers/M-element-multiply learning-rate second-layer-error-derivatives))))))))
                     weights observations)))))